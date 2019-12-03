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

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian
    """
    # z = torch.distributions.normal.Normal(m, torch.sqrt(v)).rsample()
    z = m + torch.sqrt(v) * torch.randn_like(v)
    return z

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution
    """
    # m.shape = (full_seq_len, n_sequences, feat_dim)
    m, h = torch.split(h, h.size(dim) - 1, dim=dim)
    assert h.shape[-1] == 1 # using single var
    v = F.softplus(h) + 1e-8
    # Construct linearly increasing var through seq_len
    _scale = torch.tensor(np.linspace(1 / m.shape[0], 1, m.shape[0]), 
                            dtype=m.dtype, device=m.device)
    v = (v.transpose(0, -1) * _scale).transpose(0, -1).repeat(1, 1, m.shape[-1])
    assert m.shape == v.shape
    return m, v

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    """
    log_prob = (-torch.pow(x - m, 2) / v - torch.log(2 * np.pi * v)) / 2
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob

def log_normal_for_weights(weights, means, logvars):
    """
    weights is from a multivariate gaussian with diagnol variance
    return the loglikelihood.
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
        ll += ll1 + torch.log(pi + (1 - pi) * (ll2 - ll1).exp())

    return ll

def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

def save_model_by_name(model, global_step, only_latest=False):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if only_latest:
        ckpt_fname = 'model_final.pt'
    else:
        ckpt_fname = 'model-{:05d}.pt'.format(global_step)
    file_path = os.path.join(save_dir, ckpt_fname)
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

def plot_history(model, data, iter, obj):
    plt.figure()
    plt.plot(data)
    if obj == 'loss':
        plt.yscale('symlog')
    elif obj == 'mse':
        plt.yscale('log')
    plt.xlabel('iter')
    plt.ylabel(obj)
    plt.savefig('./logs/{}/{}.png'.format(model.name, obj))
    plt.close()

def evaluate_model(model, val_set):
    # Option 1: sample the weights of the model multiple times,
    # and get the mean of the ouput for each evaluation data point
    # Option 2: do one forward pass using the mean of the weights
    # return some_metrics
    pass

def plot_highd_traj_BBB(model, iter, full_true_traj, n_resample_weights=10):
    with torch.no_grad():
        inputs = full_true_traj[:model.n_input_steps, :, :].detach()

        for i in range(n_resample_weights):
            # not using sharpening
            pred = model.forward(inputs) # one output sample
            if i == 0:
                pred_list = pred.unsqueeze(-1)
            else:
                pred = pred.unsqueeze(-1)
                pred_list = torch.cat((pred_list, pred), dim=-1)
        mean_pred = pred_list.mean(dim=-1)
        std_pred = pred_list.std(dim=-1)

    plot_highd_traj(model, iter, full_true_traj, mean_pred, std_pred=std_pred)



def plot_highd_traj(model, iter, full_true_traj, pred_traj, std_pred=None):
    with torch.no_grad():
        input_true_traj = \
            full_true_traj[:model.n_input_steps, :, :2].cpu().detach().numpy()
        ground_truth = \
            full_true_traj[(model.n_input_steps - 1):, :, :2].cpu().detach().numpy()
        pred_traj = pred_traj.cpu().detach().numpy()
        if std_pred is not None:
            std_pred = std_pred.cpu().detach().numpy()
        fig, ax = plt.subplots()
        for i in range(full_true_traj.shape[1]):
            a, = ax.plot(input_true_traj[:, i, 0], input_true_traj[:, i, 1], 
                color='blue')
            b, = ax.plot(ground_truth[:, i, 0], ground_truth[:, i, 1], 
                color='green')
            if std_pred is not None:
                c = ax.errorbar(pred_traj[:, i, 0], pred_traj[:, i, 1], 
                    xerr=std_pred[:, i, 0], yerr=std_pred[:, i, 1],
                    color='red', capsize=2)
            else:
                c, = ax.plot(pred_traj[:, i, 0], pred_traj[:, i, 1], 
                    color='red')
            if i == 0:
                a.set_label('Input')
                b.set_label('Ground Truth')
                c.set_label('Pred Mean{}'.format(
                    ' and Std' if std_pred is not None else ''))
        plt.axis('equal')
        plt.xlabel('x')
        plt.xlim(-0.15, 1.0)
        plt.ylabel('y')
        plt.title('iter = {}'.format(iter))
        # plt.ylim(-1, 1)
        ax.legend()
        plt.savefig('./logs/{}/pred_iter={}.png'.format(model.name, iter))
        plt.close()


def test_plot(model, iter, kernel):
    """
        plot sample trajectories
    """
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
            if model.constant_var:
                var = model.pred_var * torch.ones_like(outputs)
                pred_seq = sample_gaussian(outputs, var)
            else:
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
                if model.constant_var:
                    plt.errorbar(t[model.n_input_steps:], outputs.numpy(), 
                        yerr=var.squeeze().sqrt().numpy(), capsize=2, 
                        label='Mean and const Std')
                else:
                    plt.errorbar(t[model.n_input_steps:], mean.numpy(), 
                        yerr=var.squeeze().sqrt().numpy(), capsize=2, 
                        label='Mean and Std')
            plt.xlabel('t')
            plt.ylabel('x')
        elif model.input_feat_dim == 2:
            plt.plot(given_seq[:model.n_input_steps, 0, 0].numpy(), 
                    given_seq[:model.n_input_steps, 0, 1].numpy(), label='Input')
            plt.plot(given_seq[(model.n_input_steps - 1):, 0, 0].numpy(), 
                    given_seq[(model.n_input_steps - 1):, 0, 1].numpy(), 
                    label='Ground Truth')
            plt.plot(pred_seq[:, 0, 0].numpy(), 
                     pred_seq[:, 0, 1].numpy(), label='One Prediction Sample')
            if model.likelihood_cost_form == 'gaussian':
                std = var.squeeze().sqrt().numpy()
                if model.constant_var:
                    plt.errorbar(outputs[:, 0, 0].numpy(), outputs[:, 0, 1].numpy(),
                        xerr=std[:, 0], yerr=std[:, 1], capsize=2, 
                        label='Mean and const Std')
                else:
                    plt.errorbar(mean[:, 0, 0].numpy(), mean[:, 0, 1].numpy(), 
                        xerr=std[:, 0], yerr=std[:, 1], 
                        capsize=2, label='Mean and Std')
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')

        plt.title('iter = {}'.format(iter))
        plt.legend()
        plt.savefig('./logs/{}/pred_iter={}.png'.format(model.name, iter))
        plt.close()
