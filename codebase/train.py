import torch
import torch.nn as nn
from torch import optim

import numpy as np
import tqdm

import codebase.utils as ut
import data.data_utils as data_ut

def train(model, train_data, batch_size, n_batches, device, 
            lr=1e-3,
            clip_grad=None,
            iter_max=np.inf, 
            iter_save=np.inf, 
            iter_plot=np.inf, 
            reinitialize=False,
            kernel=None):

    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss()

    # # Model
    # hidden = model.init_hidden(batch_size)

    i = 0 # i is num of gradient steps taken by end of loop iteration
    loss_list = []
    with tqdm.tqdm(total=iter_max) as pbar:
        while True:
            for batch in train_data:
                i += 1 
                optimizer.zero_grad()

                inputs = batch[:model.n_input_steps, :, :]
                targets = batch[model.n_input_steps:, :, :]
                
                # Since the data is not continued from batch to batch,
                # reinit hidden every batch. (using zeros)
                outputs = model.forward(inputs, targets=targets)
                batch_mean_nll, KL, KL_sharp = model.get_loss(outputs, targets)
                # print(batch_mean_nll, KL, KL_sharp)
                
                # # Re-weighting for minibatches
                NLL_term = batch_mean_nll * model.n_pred_steps

                # KL_term = KL 
                # KL_term = KL / batch_size
                # KL_term = KL / n_batches
                KL_term = KL / n_batches / batch_size

                loss = NLL_term + KL_term

                if model.sharpen:
                    # loss += KL_sharp / batch_size
                    loss += KL_sharp / n_batches

                loss_list.append(loss)

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                # Print progress
                if model.likelihood_cost_form == 'gaussian':
                    if model.constant_var:
                        var = model.pred_var * torch.ones_like(outputs)
                        sampled_pred = ut.sample_gaussian(outputs, var)
                    else:
                        mean, var = ut.gaussian_parameters(outputs, dim=-1)
                        sampled_pred = ut.sample_gaussian(mean, var)
                    mse_val = mse(sampled_pred, targets)
                elif model.likelihood_cost_form == 'mse':
                    mse_val = batch_mean_nll
                
                pbar.set_postfix(loss='{:.2e}'.format(loss), 
                                 mse='{:.2e}'.format(mse_val))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i, only_latest=True)
                    # ut.save_latest_model(model)

                if i % iter_plot == 0:
                    if model.input_feat_dim <= 2:
                        ut.test_plot(model, i, kernel)
                    ut.plot_log_loss(model, loss_list, i)

                if i == iter_max:
                    return

