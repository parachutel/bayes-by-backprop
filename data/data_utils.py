import os
import shutil
import sys
import glob
from random import shuffle
import pandas as pd
import numpy as np
import tqdm

import torch
from torchvision import datasets, transforms

# Path constant vars:
PROJECT_DIR_PREFIX = '/Users/shengli/Desktop/BBB'
STOCKS_DATA_DIR = PROJECT_DIR_PREFIX + \
    '/data/raw/price-volume-data-for-all-us-stocks-etfs/Stocks/*.txt'


def read_highd_data(fname, batch_size, device):
    data_dir = PROJECT_DIR_PREFIX + '/data/processed/{}.pt'.format(fname)
    data_set = torch.load(data_dir).float().to(device)
    print('\nLoading HighD dataset with seq_len = {} and batch_size = {}'.format(
        data_set.shape[0], batch_size))
    data_set = torch.split(data_set, batch_size, dim=1)
    print('Number of batches = {}\n'.format(len(data_set)))
    return list(data_set)[:-1] # discard last batch

def is_non_zero_file(fpath):  
    return True if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else False

def time_grid(start, end, sequence_len):
    return np.linspace(start, end, sequence_len)

def sinusoidal_kernel(t, input_feat_dim):
    if input_feat_dim == 1:
        # # Simple
        return np.random.randint(1, 4) * np.sin(t) * np.cos(4 * t) + (t - t[0])
    elif input_feat_dim == 2:
        # # Hard
        # std = np.random.rand() * 2
        # mean = np.random.randint(-3, 4)
        # wave_scale = np.random.randn(2) * std + mean
        # a = [wave_scale[0] * np.sin(np.random.rand() * 2 * t) + (t - t[0]),
        #      wave_scale[1] * np.cos(np.random.rand() * 2 * t) + (t - t[0])]
        # # Simple:
        a = [np.random.randint(1, 4) * np.sin(np.random.rand() * 2 * t) + (t - t[0]),
             np.random.randint(1, 4) * np.cos(np.random.rand() * 2 * t) + (t - t[0])]
        return np.transpose(np.array(a))

def dummy_data_creator(batch_size, n_batches, input_feat_dim, 
                        n_input_steps, n_pred_steps, kernel, device):

    with torch.no_grad():
        data = []
        sequence_len = n_input_steps + n_pred_steps
        for i in range(n_batches):
            batch = torch.zeros((sequence_len, batch_size, input_feat_dim), 
                                device=device)
            # Vectorize?
            for batch_item_idx in range(batch_size):
                start = np.random.randint(100000)
                t = time_grid(start, start + 20, sequence_len)
                x = kernel(t, input_feat_dim).reshape(sequence_len, -1)
                batch[:, batch_item_idx, :] = \
                    torch.tensor(x, device=device, 
                        dtype=batch.dtype, requires_grad=False)
            data.append(batch)
        print('Dummy data created!')
    return data


def segment_raw_data(data_frame, full_seq_len):
    n_records = data_frame.shape[0]
    n_partitions = int(n_records / full_seq_len)
    assert n_partitions > 0
    sequences = []
    for i in range(n_partitions):
        seq = data_frame.iloc[i * full_seq_len : (i + 1) * full_seq_len, :]
        sequences.append(seq.values)

    # returns a list of sequences with each's length = full_seq_len
    return sequences

# batch_size, n_input_steps, n_pred_steps, device
def load_stocks_data(batch_size=128, full_seq_len=120, device='cpu'):
    input_feat_dim = 4
    # Check if processed stocks data exists
    processed_training_data_file_name = PROJECT_DIR_PREFIX + \
        '/data/processed/stocks_training_seqLen={}_batchSize={}.pt'.\
        format(full_seq_len, batch_size)
    processed_validation_data_file_name = PROJECT_DIR_PREFIX + \
        '/data/processed/stocks_validation_seqlen={}_batchSize={}.pt'.\
        format(full_seq_len, batch_size)
    
    if os.path.exists(processed_training_data_file_name) \
        and os.path.exists(processed_validation_data_file_name):
        # Load from stored file
        print('Found processed data, loaded.')
        training_set = torch.load(processed_training_data_file_name)
        validation_set = torch.load(processed_validation_data_file_name)
        return training_set, validation_set

    # Otherwise load and process data from raw:
    print('Loading historical stocks data...')
    
    # Load file names to list
    file_paths = glob.glob(STOCKS_DATA_DIR)

    # Shuffle file name list
    shuffle(file_paths)
    
    raw_sequences = []
    with tqdm.tqdm(total=len(file_paths)) as pbar:
        for filename in file_paths:
            pbar.update(1)
            # Skip empty file
            if is_non_zero_file(filename):
                df = pd.read_csv(filename, sep=',')

                # Skip small files without enough length
                if df.shape[0] < full_seq_len or df.shape[1] != 7:
                    continue

                df = df.drop(columns=['Date', 'OpenInt', 'Volume'])
                seq_values = segment_raw_data(df, full_seq_len)
                raw_sequences.extend(seq_values)

    n_sequences = len(raw_sequences)

    print('Stocks data loaded! Containing {} sequences with length {}.'.format(
        n_sequences, full_seq_len))

    print('Processing stocks data to batches...')

    shuffle(raw_sequences)


    n_batches = int(n_sequences / batch_size)
    i_seq = 0
    training_set_split = 0.95
    n_batches_training = int(training_set_split * n_batches)
    training_set = []
    validation_set = []
    with torch.no_grad():
        with tqdm.tqdm(total=n_batches) as pbar:
            for i_batch in range(n_batches):
                pbar.update(1)
                batch = torch.zeros((full_seq_len, batch_size, input_feat_dim), 
                                    device=device)
                for batch_item_idx in range(batch_size):
                    assert i_seq < n_sequences
                    x = torch.tensor(raw_sequences[i_seq], device=device, 
                            dtype=batch.dtype, requires_grad=False)
                    batch[:, batch_item_idx, :] = x
                    i_seq += 1
                if i_batch <= n_batches_training:
                    training_set.append(batch)
                else:
                    validation_set.append(batch)
    print('{} batches are prepared, with batch size {}'.format(n_batches, batch_size))
    print('{} batches are in training set'.format(n_batches_training))
    print('{} batches are in validation set'.format(n_batches - n_batches_training))

    torch.save(training_set, processed_training_data_file_name)
    torch.save(validation_set, processed_validation_data_file_name)

    return training_set, validation_set




