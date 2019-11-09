import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import codebase.utils as ut

import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class TimeSeriesPredModel(nn.Module):
    """
    A prediction architecture with configurable input and output secquence lengths
    """
    def __init__(self, 
        input_feat_dim,
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

        self.decoder = nn.Linear(self.hidden_feat_dim, self.input_feat_dim)

        
    def forward(self, x):
        if self.task_mode == 'async-many-to-many':
            # x.shape = (seq_len, batch, input_size)
            assert len(x) == self.n_input_steps
            # Zero-padding the time-steps to 
            x = self.pad_input_sequence(x)
            # h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
            # c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
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
        output = output[self.n_input_steps:, :, :]
        targets = full_len_seq[self.n_input_steps:, :, :]
        loss = self.criterion(output, targets)
        return loss
        


def train(model, data, device, tqdm,
          iter_max=np.inf, iter_save=np.inf, iter_plot=np.inf, 
          model_name='model', reinitialize=False):
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
                    test_plot_sin(model, i)
                    plot_log_loss(model, loss_list, i)

                if i == iter_max:
                    return

def sinusoidal_function(t):
    return 2 * np.sin(t) * np.cos(4 * t) + (t - t[0])

def time_grid(start, sequence_len):
    return np.linspace(start, start + 20, sequence_len)

def dummy_data_creator(batch_size, n_batches, n_input_steps, 
                            n_pred_steps, device):
    """
    1-D data generator
    """
    with torch.no_grad():
        data = []
        input_size = 1
        sequence_len = n_input_steps + n_pred_steps
        for i in range(n_batches):
            batch = torch.zeros((sequence_len, batch_size, input_size), device=device)
            for batch_item_idx in range(batch_size):
                start = np.random.randint(100000)
                t = time_grid(start, sequence_len)
                x = sinusoidal_function(t).reshape(sequence_len, -1)
                batch[:, batch_item_idx, :] = \
                    torch.tensor(x, device=device, 
                        dtype=batch.dtype, requires_grad=False)
            data.append(batch)
        print('dummy data loaded!')
    return data

def test_plot_sin(model, iter):
    """
    1-D data plotter
    """
    with torch.no_grad():
        sequence_len = model.n_input_steps + model.n_pred_steps
        start = np.random.randint(1000)
        t = time_grid(start, sequence_len)
        # batch_size = 1
        given_seq = torch.tensor(sinusoidal_function(t), device=model.device, 
            dtype=torch.float32, requires_grad=False).reshape(sequence_len, 1, -1)
        pred_seq = model.forward(given_seq[:model.n_input_steps, :, :])
        pred_seq = pred_seq[model.n_input_steps:, :, :]

        plt.figure()
        plt.plot(t, given_seq[:, 0, 0].numpy(), label='Ground Truth')
        plt.plot(t[model.n_input_steps:], pred_seq[:, 0, 0].numpy(), 
            label='Prediction')
        # plt.show()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('iter = {}'.format(iter))
        plt.legend()
        plt.savefig('model={}_iter={}.png'.format(model.name, iter))
        plt.clf()

def plot_log_loss(model, loss, iter):
    plt.figure()
    plt.plot(np.log(loss))
    plt.xlabel('iter')
    plt.ylabel('log-loss')
    plt.savefig('loss_model={}.png'.format(model.name))
    plt.clf()


if __name__ == '__main__':
    model_name = 'test_lstm'
    print('Model name:', model_name)

    # Data
    batch_size = 50
    n_batches = 1000
    n_input_steps = 50
    n_pred_steps = 20
    input_feat_dim=1
    # Network
    hidden_feat_dim=50
    # Train settings
    iter_max = 30000
    iter_save = np.inf
    iter_plot = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dummy_data_creator(
        batch_size=batch_size, 
        n_batches=n_batches, 
        n_input_steps=n_input_steps, 
        n_pred_steps=n_pred_steps,
        device=device)

    model = TimeSeriesPredModel(
        input_feat_dim=input_feat_dim,
        hidden_feat_dim=hidden_feat_dim,
        n_input_steps=n_input_steps,
        n_pred_steps=n_pred_steps,
        name=model_name,
        device=device).to(device)

    train(model=model,
          data=data,
          device=device,
          tqdm=tqdm.tqdm,
          iter_plot=iter_plot,
          iter_max=iter_max,
          iter_save=iter_save)

