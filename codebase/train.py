import torch
from torch import optim

import numpy as np
import tqdm

import codebase.utils as ut
import data.data_utils as data_ut

def train(model, train_data, batch_size, n_batches, device, 
            kernel=None,
            lr=1e-3,
            clip_grad=5,
            iter_max=np.inf, 
            iter_save=np.inf, 
            iter_plot=np.inf, 
            reinitialize=False):

    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Model
    hidden = model.init_hidden(batch_size)

    i = 0 
    loss_list = []
    with tqdm.tqdm(total=iter_max) as pbar:
        while True:
            for batch in train_data:
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                inputs = batch[:model.n_input_steps, :, :]
                targets = batch[model.n_input_steps:, :, :]
                outputs, hidden = model.forward(inputs, hidden, targets)
                NLL, KL, KL_sharp = model.get_loss(outputs, targets)
                
                # # Re-weighting for minibatches
                # Here NLL is already the sum over the minibatch
                NLL_term = NLL
                # KL(q|p) / BC
                KL_term = KL / batch_size / n_batches
                loss = NLL_term + KL_term
                if model.sharpen:
                    loss += KL_sharp / n_batches

                loss_list.append(loss)

                # `clip_grad_norm` helps prevent the exploding 
                # gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                loss.backward(retain_graph=True)
                # loss.backward()
                optimizer.step()

                pbar.set_postfix(loss='{:.2e}'.format(loss[0]))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i % iter_plot == 0:
                    # if model.input_feat_dim <= 2:
                    #     test_plot(model, i, kernel)
                    ut.plot_log_loss(model, loss_list, i)

                if i == iter_max:
                    return

