import torch
import torch.nn as nn
from torch import optim

import numpy as np
import tqdm

import codebase.utils as ut
import data.data_utils as data_ut

def train(model, train_data, batch_size, n_batches, device, 
            lr=2,
            clip_grad=5,
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
                # Since the data is not continued from batch to batch
                # reinit hidden every batch
                # hidden = model.init_hidden(batch_size)
                outputs = model.forward(inputs, targets=targets)
                batch_mean_nll, KL, KL_sharp = model.get_loss(outputs, targets)
                
                # # Re-weighting for minibatches
                # rescale to the mean nll of each sequence?
                NLL_term = batch_mean_nll * model.n_pred_steps * batch_size

                # KL_term = KL / batch_size / n_batches
                # KL_term = KL 
                # KL_term = KL / batch_size
                KL_term = KL / n_batches

                loss = NLL_term + KL_term
                # print(NLL_term, KL_term)

                if model.sharpen:
                    loss += KL_sharp / batch_size
                    # loss += KL_sharp / n_batches

                loss_list.append(loss)

                # `clip_grad_norm` helps prevent the exploding 
                # gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                # Print progress
                if model.likelihood_cost_form == 'gaussian':
                    mean, var = ut.gaussian_parameters(outputs, dim=-1)
                    sampled_pred = ut.sample_gaussian(mean, var)
                    mse_val = mse(sampled_pred, targets)
                elif model.likelihood_cost_form == 'mse':
                    mse_val = batch_mean_nll
                pbar.set_postfix(loss='{:.2e}'.format(loss[0]), 
                                 mse='{:.2e}'.format(mse_val))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i % iter_plot == 0:
                    if model.input_feat_dim <= 2:
                        ut.test_plot(model, i, kernel)
                    ut.plot_log_loss(model, loss_list, i)

                if i == iter_max:
                    return

