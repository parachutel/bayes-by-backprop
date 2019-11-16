import torch.nn as nn
import codebase.utils as ut
from codebase.models.nns.BBBLinear import BBBLinear
from codebase.models.nns.BBBRNN import BBBRNN
from torch.autograd import Variable
import torch

class BBBTimeSeriesPredModel(nn.Module):
    """
    A time series prediction architecture with configurable input 
    and output secquence lengths
    """
    def __init__(self, 
        input_feat_dim,
        pred_feat_dim,
        hidden_feat_dim,
        n_input_steps,
        n_pred_steps,
        device,
        num_rnn_layers=1,
        dropout=0,
        BBB=True,
        sharpen=False,
        training=True,
        likelihood_cost_form='gaussian',
        constant_var=True,
        task_mode='async-many-to-many',
        rnn_cell_type='LSTM',
        name='model',
        *args, **kwargs):

        super(BBBTimeSeriesPredModel, self).__init__()

        self.device = device
        self.name = name
        self.input_feat_dim = input_feat_dim
        self.pred_feat_dim = pred_feat_dim
        self.hidden_feat_dim = hidden_feat_dim
        self.n_input_steps = n_input_steps
        self.n_pred_steps = n_pred_steps
        self.full_seq_len = n_input_steps + n_pred_steps
        self.num_rnn_layers = num_rnn_layers
        self.training = training
        self.dropout = dropout
        self.BBB = BBB
        self.sharpen = sharpen
        self.task_mode = task_mode
        self.rnn_cell_type = rnn_cell_type
        self.likelihood_cost_form = likelihood_cost_form
        self.constant_var = constant_var
        self.pred_var = 0.001 # auxiliary parameter for evaluating prediction prob
        self.mse_fn = nn.MSELoss()

        # Build network
        # NOTE! Not using an encoder to process input!
        # Might should add an encoder for high dim input
        # RNN:
        if self.rnn_cell_type == 'LSTM':
            self.rnn = BBBRNN(rnn_cell_type, sharpen, input_feat_dim, 
                hidden_feat_dim, num_rnn_layers, dropout=dropout, BBB=self.BBB,
                *args, **kwargs)
        # One Layer Linear decoder:
        if self.likelihood_cost_form == 'gaussian':
            dim_scale = 1 if self.constant_var else 2
            self.decoder = BBBLinear(self.hidden_feat_dim, 
                self.pred_feat_dim * dim_scale, BBB=self.BBB, *args, **kwargs)
        elif self.likelihood_cost_form == 'mse':
            self.decoder = BBBLinear(self.hidden_feat_dim, self.pred_feat_dim, 
                                 BBB=self.BBB, *args, **kwargs)

        self.layers = [self.rnn, self.decoder]

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_cell_type == 'LSTM':
            return (Variable(weight.new(self.num_rnn_layers, batch_size, 
                                        self.hidden_feat_dim).zero_()),
                    Variable(weight.new(self.num_rnn_layers, batch_size, 
                                        self.hidden_feat_dim).zero_()))
        else:
            return Variable(weight.new(self.num_rnn_layers, batch_size, 
                                        self.hidden_feat_dim).zero_())

    def pad_input_sequence(self, x):
        """
        :param x: [n_input_steps, bsz, inp_dim]
        """
        zero_pad = torch.zeros(self.n_pred_steps, x.shape[1], 
            self.input_feat_dim).to(self.device)
        return torch.cat((x, zero_pad), dim=0)

    def forward(self, inputs, targets=None):
        """
        :param input: [n_input_steps, bsz, inp_dim]
        :return: [n_pred_steps, bsz, inp_dim]
        """
        if self.task_mode == 'async-many-to-many':
            assert len(inputs) == self.n_input_steps
            # Zero-padding the time-steps
            inputs = self.pad_input_sequence(inputs)
            encoded_outputs, _ = self.rnn(inputs)
            outputs = self.decoder(encoded_outputs)
            outputs = outputs[self.n_input_steps:, :, :]

        # Posterior sharpening
        if self.BBB and self.sharpen and self.training:
            # Compute the data-related loss
            NLL = self.get_nll(outputs, targets)
            gradients = torch.autograd.grad(
                outputs=NLL, 
                inputs=self.rnn.sampled_weights, 
                grad_outputs=torch.ones(NLL.size(), device=self.device), 
                create_graph=True, 
                retain_graph=True, 
                only_inputs=True)
            # Then do the forward pass again with sharpening:
            encoded_output, _ = self.rnn(inputs, grads=gradients)
            outputs = self.decoder(encoded_output)
            if self.task_mode == 'async-many-to-many':     
                outputs = outputs[self.n_input_steps:, :, :]

        return outputs

    def get_nll(self, outputs, targets):
        """
        :return: negative log-likelihood of a minibatch
        """
        if self.likelihood_cost_form == 'mse':
            # This method is not validated
            return self.mse_fn(outputs, targets)
        elif self.likelihood_cost_form == 'gaussian':
            if not self.constant_var:
                mean, var = ut.gaussian_parameters(outputs, dim=-1)
                return -torch.mean(ut.log_normal(targets, mean, var))
            else:
                var = self.pred_var * torch.ones_like(outputs)
                return -torch.mean(ut.log_normal(targets, outputs, var))

    def get_loss(self, output, targets):
        """
        ! un-scaled !
        return:
            NLL: NLL is averaged over n_pred_steps and batch_size
            KL: KL is the original scale KL
        """
        # Negative log-likelihood cost
        NLL = self.get_nll(output, targets)

        # KL: complexity cost
        if self.BBB:
            KL = torch.zeros(1, device=self.device)
            for layer in self.layers:
                if layer.BBB:
                    KL += layer.get_kl()
            KL = KL.squeeze()
        else:
            KL = 0.

        if self.sharpen:
            KL_sharp = self.rnn.get_kl_sharpening()
        else:
            KL_sharp = 0.

        return NLL, KL, KL_sharp

