import torch
import numpy as np
from codebase.models.BBBTimeSeriesPredModel import BBBTimeSeriesPredModel
from codebase.train import train
import data.data_utils as data_ut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



batch_size = 50
n_batches = 100 # used by dummy data
n_input_steps = 50
n_pred_steps = 20
input_feat_dim = 2 # 4 for stocks data
pred_feat_dim = 2 # 1 for stocks data
# Network
hidden_feat_dim = 80

pi = 0.3
std1 = np.exp(1) 
std2 = np.exp(-6) 
gpu = False
BBB = True

iter_max = 50000
training = True
sharpen = True

full_seq_len = n_input_steps + n_pred_steps


model = BBBTimeSeriesPredModel(
        pi=pi,
        std1=std1,
        std2=std2,
        gpu=gpu,
        BBB=BBB,
        training=training,
        sharpen=sharpen,
        input_feat_dim=input_feat_dim,
        pred_feat_dim = pred_feat_dim,
        hidden_feat_dim=hidden_feat_dim,
        n_input_steps=n_input_steps,
        n_pred_steps=n_pred_steps,
        name='test_BBBRNN',
        device=device).to(device)

# np.random.seed(1234)
# inputs = torch.Tensor(np.random.rand(n_input_steps, batch_size, input_feat_dim))
# hidden = model.init_hidden(batch_size)
# targets = torch.Tensor(np.random.rand(n_pred_steps, batch_size, pred_feat_dim))
# outputs, hidden = model.forward(inputs, hidden, targets)
# assert outputs.shape == torch.Size([n_pred_steps, batch_size, 2 * pred_feat_dim])
# print(outputs.shape)
# loss = model.get_loss(outputs, targets)
# print(loss)

dummy_training_set = data_ut.dummy_data_creator(
        batch_size=batch_size, 
        n_batches=n_batches, 
        input_feat_dim=input_feat_dim,
        n_input_steps=n_input_steps, 
        n_pred_steps=n_pred_steps,
        kernel=data_ut.sinusoidal_kernel,
        device=device)

train(model, dummy_training_set, batch_size, n_batches, device, 
        kernel=data_ut.sinusoidal_kernel,
        lr=1e-3,
        clip_grad=5,
        iter_max=iter_max, 
        iter_save=np.inf, 
        iter_plot=np.inf, 
        reinitialize=False)

