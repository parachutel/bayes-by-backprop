import numpy as np
import torch
import scipy.stats

import codebase.utils as ut

# test ut.log_scale_gaussian_mix_prior
pi = 0.9
std1 = 1
std2 = np.e ** (-6)

w1 = np.random.rand(10, 6)
w2 = np.random.rand(7, 2)
weights = [torch.Tensor(w1), torch.Tensor(w2)]


log_p_my = ut.log_scale_gaussian_mix_prior(weights, pi, std1, std2)

print(log_p_my)

log_prob = 0
for w in [w1, w2]:
    for row in w:
        for element in row:
            p1 = scipy.stats.norm(0, std1).pdf(element)
            p2 = scipy.stats.norm(0, std2).pdf(element)
            log_prob += np.log(pi * p1 + (1 - pi) * p2)

print(log_prob)
