import torch.nn as nn
import codebase.utils as ut
from codebase.models.nns.BBBLinear import BBBLinear
from torch.autograd import Variable
import torch

class BBBTimeSeriesPredModel_FF(nn.Module):

    def __init__(self, 
        input_feat_dim,
        pred_feat_dim,
        hidden_feat_dim,
        n_input_steps,
        n_pred_steps,
        device,
        num_hidden_layers=2,
        dropout=0,
        BBB=True,
        sharpen=False,
        training=True,
        likelihood_cost_form='gaussian',
        constant_var=False,
        task_mode='async-many-to-many',
        name='model',
        *args, **kwargs):

        super(BBBTimeSeriesPredModel_FF, self).__init__()
        self.rnn_cell_type = 'FF' # for use in train function
        self.name = name
        self.device = device
        self.constant_var = constant_var
        self.BBB = BBB
        self.sharpen = sharpen
        self.training = training
        self.likelihood_cost_form = likelihood_cost_form

        # input size = [batch_size, input_size]
        # input_size = input_seq_len * input_feat_dim (50 * 4)
        # output_size = output_seq_len * output_feat_dim + var_dim (20 * 2 + 1)
        self.input_feat_dim = input_feat_dim
        self.pred_feat_dim = pred_feat_dim
        self.hidden_feat_dim = hidden_feat_dim
        self.n_input_steps = n_input_steps
        self.n_pred_steps = n_pred_steps
        self.input_size = self.n_input_steps * self.input_feat_dim
        self.output_size = self.n_pred_steps * self.pred_feat_dim
        
        if not self.constant_var:
            self.output_size += self.pred_feat_dim
        else:
            self.pred_var = 0.001 # auxiliary parameter for evaluating prediction prob

        # Feedforward architecture
        self.layers = \
            [BBBLinear(self.input_size, hidden_feat_dim, BBB=self.BBB, *args, **kwargs)]
        
        for i in range(num_hidden_layers):
            self.layers.append(nn.ELU())
            next_feat_dim = hidden_feat_dim
            if i+1 == num_hidden_layers:
                next_feat_dim = self.output_size
            self.layers.append(
                BBBLinear(hidden_feat_dim, next_feat_dim, BBB=self.BBB, *args, **kwargs))
            
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, inputs, targets=None):
        # work flow:
        # input (input_seq_len, batch_size, input_feat_dim)
        # reshape: input (batch_size, input_seq_len * input_feat_dim)
        input_seq_len, batch_size, input_feat_dim = inputs.shape
        inputs_reshape = inputs.transpose(0,1).reshape(batch_size,-1)
        
        # output = forward(input)
        outputs = self.net(inputs_reshape)
        
        # output (batch_size, output_seq_len * output_feat_dim)
        # reshape: output (output_seq_len, batch_size, output_feat_dim)        
        # compute nll and loss with reshaped output
        return outputs.reshape(batch_size,-1,self.pred_feat_dim).transpose(0,1)

    def get_nll(self, outputs, targets):
        """
        :return: negative log-likelihood of a minibatch
        """
        if not self.constant_var:
            mean, var = ut.gaussian_parameters_ff(outputs, dim=0)
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
            raise Exception('sharpening not implemented for feed forward')
            #KL_sharp = self.rnn.get_kl_sharpening()
        else:
            KL_sharp = 0.

        return NLL, KL, KL_sharp

