import torch.nn as nn
import torch
import math
from codebase.models.nn.BBBLayer import BBBLayer

class BBBLinear(BBBLayer):
    """
    adapted from torch.nn.Linear
    with Gaussian mixture as prior
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        super(BBBLinear, self).__init__(*args, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        if self.BBB is True:
            self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        # used for KL
        self.means = [self.weight_mean, self.bias_mean]
        if self.BBB is True:
            self.logvars = [self.weight_logvar, self.bias_logvar]

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mean.size(1))
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        if self.BBB is True:
            for logvar in self.logvars:
                logvar.data.fill_(logvar_init)

    def forward(self, inputs):
        if self.training and self.BBB is True:
            # if use BBB and it is training
            self.sample()
            weight = self.sampled_weights[0]
            bias = self.sampled_weights[1]
        else:
            # use only mean for testing or non BBB
            weight = self.weight_mean
            bias = self.bias_mean
        return nn.functional.linear(inputs, weight, bias)