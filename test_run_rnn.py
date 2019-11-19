import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import codebase.utils as ut
import data.data_utils as data_ut

import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class TimeSeriesPredModel(nn.Module):
    """
    A prediction architecture with configurable input and output secquence lengths
    """
    def __init__(self, 
        input_feat_dim,
        pred_feat_dim,
        hidden_feat_dim,
        n_input_steps,
        n_pred_steps,
        device,
        task_mode='async-many-to-many',
        rnn_cell_type='LSTM',
        name='model'
        ):
        super(TimeSeriesPredModel, self).__init__()

        self.device = device
        self.name = name
        self.input_feat_dim = input_feat_dim
        self.pred_feat_dim = pred_feat_dim
        self.hidden_feat_dim = hidden_feat_dim
        self.n_input_steps = n_input_steps
        self.n_pred_steps = n_pred_steps
        self.task_mode = task_mode
        self.rnn_cell_type = rnn_cell_type
        self.criterion = nn.MSELoss()

        # Build network
        if self.rnn_cell_type == 'LSTM':
            # Default one layer
            self.rnn = nn.LSTM(self.input_feat_dim, self.hidden_feat_dim)

        self.decoder = nn.Linear(self.hidden_feat_dim, self.pred_feat_dim)


        
    def forward(self, x):
        if self.task_mode == 'async-many-to-many':
            # x.shape = (seq_len, batch, input_size)
            assert len(x) == self.n_input_steps
            # Zero-padding the time-steps to 
            x = self.pad_input_sequence(x)
            encoded_output, hidden = self.rnn(x)
            output = self.decoder(encoded_output)
        return output

    def pad_input_sequence(self, x):
        zero_pad = torch.zeros(self.n_pred_steps, x.shape[1], self.input_feat_dim)
        return torch.cat((x, zero_pad), dim=0)


    def loss(self, full_len_seq):
        # full_len_seq = input x
        # Use last n_pred_steps in sequence
        # Segmenting the input sequence
        x = full_len_seq[:self.n_input_steps, :, :]
        output = self.forward(x)
        if self.name == 'test_lstm_stocks':
            output = output[self.n_input_steps:, :, :]
            # Specially tailored
            # Open    High     Low   Close
            # Target feature = High
            targets = full_len_seq[self.n_input_steps:, :, 1:2]
        else:
            output = output[self.n_input_steps:, :, :]
            targets = full_len_seq[self.n_input_steps:, :, :]
        loss = self.criterion(output, targets)
        return loss
        


def train(model, data, device, tqdm, kernel,
          iter_max=np.inf, iter_save=np.inf, iter_plot=np.inf, 
          reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    i = 0
    loss_list = []
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch in data:
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                loss = model.loss(batch)
                loss_list.append(loss)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i % iter_plot == 0:
                    if model.input_feat_dim <= 2:
                        test_plot(model, i, kernel)
                    ut.plot_log_loss(model, loss_list, i)

                if i == iter_max:
                    return


def test_plot(model, iter, kernel):
    with torch.no_grad():
        sequence_len = model.n_input_steps + model.n_pred_steps
        start = np.random.randint(1000)
        t = data_ut.time_grid(start, start + 20, sequence_len)
        # batch_size = 1
        given_seq = torch.tensor(kernel(t, model.input_feat_dim), device=model.device, 
            dtype=torch.float32, requires_grad=False).reshape(sequence_len, 1, -1)
        pred_seq = model.forward(given_seq[:model.n_input_steps, :, :])
        pred_seq = pred_seq[model.n_input_steps:, :, :]

        plt.figure()
        if model.input_feat_dim == 1:
            plt.plot(t, given_seq[:, 0, 0].numpy(), label='Ground Truth')
            plt.plot(t[model.n_input_steps:], pred_seq[:, 0, 0].numpy(), 
                label='Prediction')
            plt.xlabel('t')
            plt.ylabel('x')
        elif model.input_feat_dim == 2:
            plt.plot(given_seq[:model.n_input_steps, 0, 0].numpy(), 
                    given_seq[:model.n_input_steps, 0, 1].numpy(), label='Input')
            plt.plot(given_seq[(model.n_input_steps - 1):, 0, 0].numpy(), 
                    given_seq[(model.n_input_steps - 1):, 0, 1].numpy(), label='Ground Truth')
            plt.plot(pred_seq[:, 0, 0].numpy(), 
                    pred_seq[:, 0, 1].numpy(), label='Prediction')
            plt.xlabel('x')
            plt.ylabel('y')

        plt.title('iter = {}'.format(iter))
        plt.legend()
        plt.savefig('./logs/{}/pred_iter={}.png'.format(model.name, iter))
        plt.close()


def sinusoidal_kernel(t, input_feat_dim):
    if input_feat_dim == 1:
        # # Simple
        return np.random.randint(1, 4) * np.sin(t) * np.cos(4 * t) + (t - t[0])
    elif input_feat_dim == 2:
        # # Hard
        std = np.random.rand() * 2
        mean = np.random.randint(-3, 4)
        wave_scale = np.random.randn(2) * std + mean
        a = [wave_scale[0] * np.sin(np.random.rand() * 2 * t) + (t - t[0]),
             wave_scale[1] * np.cos(np.random.rand() * 2 * t) + (t - t[0])]
        # # Simple:
        # a = [np.random.randint(1, 4) * np.sin(np.random.rand() * 2 * t) + (t - t[0]),
        #      np.random.randint(1, 4) * np.cos(np.random.rand() * 2 * t) + (t - t[0])]
        return np.transpose(np.array(a))

if __name__ == '__main__':
    run = 1
    model_name = 'test_lstm_2d_run={}'.format(run)
    print('Model name:', model_name)

    # Data
    batch_size = 80
    n_batches = 2000 # used by dummy data
    n_input_steps = 50
    n_pred_steps = 20
    input_feat_dim = 2 # 4 for stocks data
    pred_feat_dim = 2 # 1 for stocks data
    # Network
    hidden_feat_dim = 80
    # Train settings
    iter_max = 80000
    iter_save = np.inf # Not saving models for now
    iter_plot = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_seq_len = n_input_steps + n_pred_steps
    
    # stocks_training_set, val_set = data_ut.load_stocks_data(
    #     batch_size=batch_size, 
    #     full_seq_len=full_seq_len, 
    #     device=device)

    dummy_training_set = data_ut.dummy_data_creator(
        batch_size=batch_size, 
        n_batches=n_batches, 
        input_feat_dim=input_feat_dim,
        n_input_steps=n_input_steps, 
        n_pred_steps=n_pred_steps,
        kernel=sinusoidal_kernel,
        device=device)

    model = TimeSeriesPredModel(
        input_feat_dim=input_feat_dim,
        pred_feat_dim = pred_feat_dim,
        hidden_feat_dim=hidden_feat_dim,
        n_input_steps=n_input_steps,
        n_pred_steps=n_pred_steps,
        name=model_name,
        device=device).to(device)

    ut.prepare_dirs(model_name, overwrite_existing=True)

    train(model=model,
          data=dummy_training_set, # Change
          device=device,
          tqdm=tqdm.tqdm,
          kernel=sinusoidal_kernel,
          iter_plot=iter_plot,
          iter_max=iter_max,
          iter_save=iter_save)

