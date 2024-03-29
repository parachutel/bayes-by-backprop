import torch
import torch.nn as nn
from torch import optim

import numpy as np
import tqdm
import random
# import psutil

import codebase.utils as ut
import data.data_utils as data_ut

def train(model, train_data, batch_size, n_batches, 
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
    mse_list = []
    with tqdm.tqdm(total=iter_max) as pbar:
        while True:
            for batch in train_data:
                i += 1 
                # print(psutil.virtual_memory())
                optimizer.zero_grad()

                inputs = batch[:model.n_input_steps, :, :]
                targets = batch[model.n_input_steps:, :, :2]
                
                # Since the data is not continued from batch to batch,
                # reinit hidden every batch. (using zeros)
                outputs = model.forward(inputs, targets=targets)
                batch_mean_nll, KL, KL_sharp = model.get_loss(outputs, targets)
                # print(batch_mean_nll, KL, KL_sharp)
                
                # # Re-weighting for minibatches
                NLL_term = batch_mean_nll * model.n_pred_steps

                # Here B = n_batchs, C = 1 (since each sequence is complete)
                KL_term = KL / n_batches

                loss = NLL_term + KL_term

                if model.sharpen:
                    KL_sharp /= n_batches
                    loss += KL_sharp

                loss_list.append(loss.cpu().detach())

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # Print progress
                if model.likelihood_cost_form == 'gaussian':
                    if model.constant_var:
                        mse_val = mse(outputs, targets) * model.n_pred_steps
                    else:
                        if model.rnn_cell_type == 'FF':
                            mean, var = ut.gaussian_parameters_ff(outputs, dim=0)

                        else:
                            mean, var = ut.gaussian_parameters(outputs, dim=-1)
                        mse_val = mse(mean, targets) * model.n_pred_steps

                elif model.likelihood_cost_form == 'mse':
                    mse_val = batch_mean_nll * model.n_pred_steps
                    
                mse_list.append(mse_val.cpu().detach())

                if i % iter_plot == 0:
                    with torch.no_grad():
                        model.eval()
                        if model.input_feat_dim <= 2:
                            ut.test_plot(model, i, kernel)

                        elif model.input_feat_dim == 4:
                            rand_idx = random.sample(range(batch.shape[1]), 4)
                            full_true_traj = batch[:, rand_idx, :]
                            if not model.BBB:
                                if model.constant_var:
                                    pred_traj = outputs[:, rand_idx, :]
                                    std_pred = None
                                else:
                                    pred_traj = mean[:, rand_idx, :]
                                    std_pred = var.sqrt()

                                ut.plot_highd_traj(model, i, full_true_traj, 
                                    pred_traj, std_pred=std_pred)
                            else:
                                # resample a few forward passes
                                ut.plot_highd_traj_BBB(model, i, full_true_traj, 
                                                        n_resample_weights=10)
    
                        ut.plot_history(model, loss_list, i, obj='loss')
                        ut.plot_history(model, mse_list, i, obj='mse')
                        model.train()

                
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(loss='{:.2e}'.format(loss), 
                                 mse='{:.2e}'.format(mse_val))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i, only_latest=True)

                if i == iter_max:
                    return

