import torch.nn as nn
import codebase.utils as ut
from codebase.models.nns.BBBLinear import BBBLinear
from codebase.models.nns.BBBRNN import BBBRNN
from torch.autograd import Variable
import torch

class BBBTimeSeriesPredModel_FF(nn.Module):

    def __init__(self, 
        input_feat_dim,
        pred_feat_dim,
        n_input_steps,
        n_pred_steps,
        device,
        BBB=True,
        sharpen=False,
        training=True,
        constant_var=False,
        name='model',
        *args, **kwargs):

        super(BBBTimeSeriesPredModel_FF, self).__init__()
        self.device = device
        self.constant_var = constant_var
        self.BBB = BBB
        self.sharpen = sharpen

        # input size = [batch_size, input_size]
        # input_size = input_seq_len * input_feat_dim (50 * 4)
        # output_size = output_seq_len * output_feat_dim + var_dim (20 * 2 + 1)
        self.input_size = self.n_input_steps * self.input_feat_dim
        self.output_size = self.n_pred_steps * self.pred_feat_dim
        if not self.constant_var:
            self.output_size += 1
        else:
            self.pred_var = 0.001 # auxiliary parameter for evaluating prediction prob

        # Feedforward architecture
        l0 = nn.BBBLinear(self.input_size, 500, BBB=self.BBB, *args, **kwargs)
        l1 = nn.BBBLinear(500, 500, BBB=self.BBB, *args, **kwargs)
        l2 = nn.BBBLinear(500, self.output_size, BBB=self.BBB, *args, **kwargs)

        self.layers = [l0, l1, l2]
        self.net = nn.Sequential(l0, nn.ELU(), l1, nn.ELU(), l2)

        # work flow:
        # input (input_seq_len, batch_size, input_feat_dim)
        # reshape: input (batch_size, input_seq_len * input_feat_dim)
        # output = forward(input)
        # output (batch_size, output_seq_len * output_feat_dim)
        # reshape: output (output_seq_len, batch_size, output_feat_dim)
        # compute nll and loss with reshaped output

    def get_nll(self, outputs, targets):
        """
        :return: negative log-likelihood of a minibatch
        """
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