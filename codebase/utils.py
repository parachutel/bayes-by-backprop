import torch
from torch.distributions.normal import Normal
import math

import scipy.stats
import numpy as np

def mul_var_normal(weights, means, logvars):
    """
    theta is from a multivariate gaussian with diagnol covariance
    return the loglikelihood.
    :param weights: a list of weights
    :param means: a list of means
    :param logvars: a list of logvars
    :return ll: loglikelihood sum over list
    """
    ll = 0

    for i in range(len(weights)):
        w = weights[i]
        mean = means[i]
        
        if len(logvars) > 1:
            logvar = logvars[i]
            var = logvar.exp()
        else:
            logvar = logvars[0]
            var = math.exp(logvar)

        logstd = logvar * 0.5
        ll += torch.sum(
            -((w - mean)**2)/(2*var) - logstd - math.log(math.sqrt(2*math.pi))
        )

    return ll

def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def log_scale_gaussian_mix_prior(weights, pi, std1, std2):
    """
    Computes the prior log-likelihood of weights using scale mixture prior,
    providing a heavier tail in the prior density than plain Gaussian
        P(w) = prod_j pi * N(w_j; 0, std1^2) + (1 - pi) * N(w_j; 0, std2^2)
        log(P(w)) = sum_j 

    :param weights: a list of weights, e.g. [weight, bias]
    :param pi: a scalar, weight of the large-variance Gaussian component
    :param logstd1: a scalar, hyper-param, std1 is large
    :param logstd2: a scalar, hyper-param, std2 << 1
    :return log_prob: prior log-likelihood of the weights
    """
    log_prob = 0
    for w in weights:
        # weights is a collection of weight vectors
        # w contains numerous elements
        log_prob1 = Normal(0, std1).log_prob(w)
        log_prob2 = Normal(0, std2).log_prob(w)
        max_prob = torch.max(log_prob1, log_prob2)
        # Numerically stable scaled log_sum_exp
        log_mix_prob = log_prob1 \
            + (pi + (1 - pi) * ((log_prob2 - log_prob1).exp())).log()
        log_prob += torch.sum(log_mix_prob)

    return log_prob


