import numpy as np
import torch
import scipy.stats

import codebase.utils as ut

# Test ut.log_scale_gaussian_mix_prior
pi = 0.3
std1 = 1
std2 = np.e ** (-6)

w1 = np.random.rand(10, 6)
w2 = np.random.rand(7, 2)
weights = [torch.Tensor(w1), torch.Tensor(w2)]


log_p = ut.log_scale_gaussian_mix_prior(weights, pi, std1, std2)

print(log_p)
log_prob = 0
for w in [w1, w2]:
    for row in w:
        for element in row:
            p1 = scipy.stats.norm(0, std1).pdf(element)
            p2 = scipy.stats.norm(0, std2).pdf(element)
            log_prob += np.log(pi * p1 + (1 - pi) * p2)

print(log_prob)


# Test BBBLinear
# from codebase.models.nns.BBBLinear import BBBLinear
# import torch.nn as nn

# test_net = nn.Sequential(
#     BBBLinear(784, 300, 
#         pi=pi, std1=std1,
#         std2=std2, BBB=True,
#         gpu=False),
#     nn.ELU(),
#     BBBLinear(300, 300, 
#         pi=pi, std1=std1,
#         std2=std2, BBB=True,
#         gpu=False),
#     nn.ELU(),
#     BBBLinear(300, 4,
#         pi=pi, std1=std1,
#         std2=std2, BBB=True,
#         gpu=False),
# )
# # Batch x Channel x Height x Width
# test_net_out = test_net(torch.Tensor(np.random.rand(128, 3, 1, 784)))
# print(test_net_out.shape)