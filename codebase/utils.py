import os
import shutil
import sys

import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal

import math
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: scalar: log probability of all the samples.
    """
    log_prob = (-torch.pow(x - m, 2) / v - torch.log(2 * np.pi * v)) / 2
    log_prob = torch.sum(log_prob)
    return log_prob

def log_normal_for_weights(weights, means, logvars):
    """
    weights is from a multivariate gaussian with diagnol variance
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
        # max_prob = torch.max(log_prob1, log_prob2)
        # Numerically stable scaled log_sum_exp
        log_mix_prob = log_prob1 \
            + (pi + (1 - pi) * ((log_prob2 - log_prob1).exp())).log()
        log_prob += torch.sum(log_mix_prob)

    return log_prob

def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_dirs(model_name, overwrite_existing=False):
    save_dir = os.path.join('checkpoints', model_name)
    log_dir = os.path.join('logs', model_name)
    if overwrite_existing:
        delete_existing(save_dir)
        delete_existing(log_dir)
    # Create dirs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)

def plot_log_loss(model, loss, iter):
    plt.figure()
    plt.plot(np.log(loss))
    plt.xlabel('iter')
    plt.ylabel('log-loss')
    plt.savefig('./logs/{}/loss.png'.format(model.name))
    plt.close()

