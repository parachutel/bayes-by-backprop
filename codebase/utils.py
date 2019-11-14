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

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    # z = torch.distributions.normal.Normal(m, torch.sqrt(v)).rsample()
    z = m + torch.sqrt(v) * torch.randn_like(v) 
    return z

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
    log_prob = torch.sum(log_prob, dim=-1)
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
    # # Viewing w_j as an elemental weight entry
    # log_prob = 0
    # for w in weights:
    #     # weights is a collection of weight vectors
    #     # w contains numerous elements
    #     log_prob1 = Normal(0, std1).log_prob(w)
    #     log_prob2 = Normal(0, std2).log_prob(w)
    #     # Numerically stable scaled log_sum_exp
    #     log_mix_prob = log_prob1 \
    #         + (pi + (1 - pi) * ((log_prob2 - log_prob1).exp())).log()
    #     log_prob += torch.sum(log_mix_prob)

    # # Viewing w_j as a weight matrix
    ll = 0
    for w in weights:
        var1 = std1 ** 2
        ll1 = torch.sum(
            -(w**2)/(2*var1) - np.log(std1) - math.log(math.sqrt(2*math.pi))
        )
        var2 = std2 ** 2
        ll2 = torch.sum(
            -(w**2)/(2*var2) - np.log(std2) - math.log(math.sqrt(2*math.pi))
        )
        # use a numerical stable one
        # ll1 + log(pi + (1-pi) exp(ll2-ll1))
        ll += ll1 + ( pi + (1-pi) * ((ll2-ll1).exp()) ).log()

    return ll

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

def test_plot(model, iter, kernel):
    import data.data_utils as data_ut
    with torch.no_grad():
        sequence_len = model.n_input_steps + model.n_pred_steps
        start = np.random.randint(1000)
        t = data_ut.time_grid(start, start + 20, sequence_len)
        # batch_size = 1
        given_seq = torch.tensor(kernel(t, model.input_feat_dim), device=model.device, 
            dtype=torch.float32, requires_grad=False).reshape(sequence_len, 1, -1)
        inputs = given_seq[:model.n_input_steps, :, :]
        outputs = model.forward(inputs)
        if model.likelihood_cost_form == 'gaussian':
            mean, var = gaussian_parameters(outputs, dim=-1)
            pred_seq = sample_gaussian(mean, var)
        elif model.likelihood_cost_form == 'mse':
            pred_seq = outputs

        plt.figure()
        if model.input_feat_dim == 1:
            plt.plot(t, given_seq[:, 0, 0].numpy(), label='Ground Truth')
            plt.plot(t[model.n_input_steps:], pred_seq[:, 0, 0].numpy(), 
                label='One Prediction Sample')
            if model.likelihood_cost_form == 'gaussian':
                plt.errorbar(t[model.n_input_steps:], mean.numpy(), 
                    yerr=var.squeeze().sqrt().numpy(), capsize=2, 
                    label='Mean and Std')
            plt.xlabel('t')
            plt.ylabel('x')
        elif model.input_feat_dim == 2:
            plt.plot(given_seq[:model.n_input_steps, 0, 0].numpy(), 
                    given_seq[:model.n_input_steps, 0, 1].numpy(), label='Input')
            plt.plot(given_seq[(model.n_input_steps - 1):, 0, 0].numpy(), 
                    given_seq[(model.n_input_steps - 1):, 0, 1].numpy(), label='Ground Truth')
            plt.plot(pred_seq[:, 0, 0].numpy(), 
                     pred_seq[:, 0, 1].numpy(), label='One Prediction Sample')
            if model.likelihood_cost_form == 'gaussian':
                plt.errorbar(mean[:, 0, 0].numpy(), mean[:, 0, 1].numpy(), 
                    xerr=var.squeeze().sqrt().numpy()[:, 0],
                    yerr=var.squeeze().sqrt().numpy()[:, 1], 
                    capsize=2, label='Mean and Std')
            plt.xlabel('x')
            plt.ylabel('y')

        plt.title('iter = {}'.format(iter))
        plt.legend()
        plt.savefig('./logs/{}/pred_iter={}.png'.format(model.name, iter))
        plt.close()
