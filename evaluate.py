import torch
from torch.utils.data import random_split
from torchsummary import summary
import argparse
import numpy as np
import random
from codebase.models.BBBTimeSeriesPredModel import BBBTimeSeriesPredModel
from codebase.models.BBBTimeSeriesPredModel_FF import BBBTimeSeriesPredModel_FF
from codebase.train import train
import codebase.utils as ut
import data.data_utils as data_ut
from tqdm import tqdm

### RWSE Functions

def rwse(model, full_true_trajs, n_samples=100):
    """ root-weighted square error (RWSE) captures 
        the deviation of a model’s probability
        mass from real-world trajectories
    """
    inputs = full_true_trajs[:model.n_input_steps, :, :].detach()
    targets = full_true_trajs[model.n_input_steps:, :, :2].detach()
    
    # tile based on number of samples
    inputs = inputs.repeat(1,n_samples,1)
    targets = targets.repeat(1,n_samples,1)
    
    if model.BBB and model.rnn_cell_type=="LSTM":
        wse = wse_bbb_rnn(model, inputs, targets)
    elif not model.BBB and model.rnn_cell_type=="LSTM":
        wse = wse_rnn(model, inputs, targets)
    elif model.BBB and model.rnn_cell_type=="FF":
        wse = wse_bbb_ff(model, inputs, targets)
    elif not model.BBB and model.rnn_cell_type=="FF":
        wse = wse_ff(model, inputs, targets)
    else:
        raise Exception('Incorrect model specified')
        
    rwse = wse.mean().sqrt()
    return rwse

def wse_bbb_rnn(model, inputs, targets):
    pred = model.forward(inputs).detach()
    if not model.constant_var:
        pred = pred[:, :, :-1]
    return ((targets - pred) ** 2).sum(-1).sum(0)

def wse_rnn(model, inputs, targets):
    pred = model.forward(inputs).detach()
    if not model.constant_var:
        mean, var = ut.gaussian_parameters(pred, dim=-1)
    else:
        mean = pred
        var = model.pred_var
    sample_trajs = ut.sample_gaussian(mean, var)
    return ((targets - sample_trajs) ** 2).sum(-1).sum(0)

def wse_bbb_ff(model,inputs,targets):
    raise Exception('Yet to formulate bbb ff')

def wse_ff(model, inputs, targets):
    pred = model.forward(inputs).detach()
    if not model.constant_var:
        mean, var = ut.gaussian_parameters_ff(pred, dim=0)
    else:
        mean = pred
        var = model.pred_var

    sample_trajs = ut.sample_gaussian(mean, var)
    return ((targets - sample_trajs) ** 2).sum(-1).sum(0)

### RMSE Functions

def rmse(model, full_true_trajs, n_samples=100):
    """ root-mean square error (RMSE) captures 
        the deviation of a model’s expected trajectory from
        mass from real-world trajectories
    """    
    inputs = full_true_trajs[:model.n_input_steps, :, :].detach()
    targets = full_true_trajs[model.n_input_steps:, :, :2].detach()
    
    if model.BBB and model.rnn_cell_type=="LSTM":
        mse = mse_bbb_rnn(model, inputs, targets, n_samples)
    elif not model.BBB and model.rnn_cell_type=="LSTM":
        mse = mse_rnn(model, inputs, targets, n_samples)
    elif model.BBB and model.rnn_cell_type=="FF":
        mse = mse_bbb_ff(model, inputs, targets, n_samples)
    elif not model.BBB and model.rnn_cell_type=="FF":
        mse = mse_ff(model, inputs, targets, n_samples)
    else:
        raise Exception('Incorrect model specified')
        
    rmse = mse.mean().sqrt()
    return rmse

def mse_bbb_rnn(model, inputs, targets, n_samples): #FIXME
    sample_tensor = torch.zeros(*targets.shape,n_samples)
    for i in range(n_samples):
        pred = model.forward(inputs).detach()
        if not model.constant_var:
            pred = pred[:, :, :-1]
        sample_tensor[:,:,:,i] = pred
    mean_pred = sample_tensor.mean(3)
    return ((targets - mean_pred) ** 2).sum(-1).sum(0)

def mse_rnn(model, inputs, targets, n_samples):
    pred = model.forward(inputs).detach()
    if not model.constant_var:
        mean, var = ut.gaussian_parameters(pred, dim=-1)
    else:
        mean = pred
        var = model.pred_var
    return ((targets - mean) ** 2).sum(-1).sum(0)

def mse_bbb_ff(model, inputs, targets, n_samples):
    raise Exception('Yet to formulate bbb ff')

def mse_ff(model, inputs, targets, n_samples):
    pred = model.forward(inputs).detach()
    if not model.constant_var:
        mean, var = ut.gaussian_parameters_ff(pred, dim=0)
    else:
        mean = pred
        var = model.pred_var
    return ((targets - mean) ** 2).sum(-1).sum(0)


### Run on Execution ::

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--dataset_name', type=str, default='highd')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--n_input_steps', type=int, default=50)
parser.add_argument('--n_pred_steps', type=int, default=20)
parser.add_argument('--input_feat_dim', type=int, default=4)
parser.add_argument('--pred_feat_dim', type=int, default=2)
# Network
parser.add_argument('--hidden_feat_dim', type=int, default=100)
# Model
parser.add_argument('--cell', type=str, default='LSTM')
parser.add_argument('--constant_var', type=int, default=0)
parser.add_argument('--BBB', type=int, default=0)
parser.add_argument('--sharpen', type=int, default=0)
parser.add_argument('--likelihood_cost_form', type=str, default='gaussian')
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--pi', type=float, default=0.25)
parser.add_argument('--logstd1', type=int, default=-1)
parser.add_argument('--logstd2', type=int, default=-6)
# Train
parser.add_argument('--clip_grad', type=int, default=5)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--training', action='store_true', default=False)

args = parser.parse_args()

std1 = np.exp(args.logstd1)
std2 = np.exp(args.logstd2)

# # automatic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = False if device == torch.device('cpu') else True

# Enforced settings:
if not args.BBB:
    args.sharpen = False
    layout = [
        ('model={:s}', args.cell),
        ('BBB={}', bool(args.BBB)),
        ('data={:s}', args.dataset_name),
        ('nlayers={:d}', args.nlayers),
        ('nhid={:d}', args.hidden_feat_dim),
        ('const_var={}', bool(args.constant_var)),
        ('dropout={:.1f}', args.dropout),
        ('clipgrad={}', str(args.clip_grad)),
        ('loss={:s}', args.likelihood_cost_form),
        ('run={:d}', args.run),
    ]
else:
    layout = [
        ('model={}', args.cell),
        ('BBB={}', bool(args.BBB)),
        ('data={:s}', args.dataset_name),
        ('nlayers={:d}', args.nlayers),
        ('nhid={:d}', args.hidden_feat_dim),
        ('const_var={}', bool(args.constant_var)),
        ('dropout={:.1f}', args.dropout),
        ('clipgrad={}', str(args.clip_grad)),
        ('loss={:s}', args.likelihood_cost_form),
        ('sharpen={}', bool(args.sharpen)),
        ('pi={:.2f}', args.pi),
        ('logstd1={:d}', args.logstd1),
        ('logstd2={:d}', args.logstd2),
        ('run={:d}', args.run),
    ]

model_name = '_'.join([t.format(v) for (t, v) in layout])

if args.cell == 'LSTM':
    model = BBBTimeSeriesPredModel(
            num_rnn_layers=args.nlayers,
            pi=args.pi,
            std1=std1,
            std2=std2,
            gpu=gpu,
            BBB=bool(args.BBB),
            training=args.training,
            sharpen=bool(args.sharpen),
            dropout=args.dropout,
            likelihood_cost_form=args.likelihood_cost_form,
            input_feat_dim=args.input_feat_dim,
            pred_feat_dim=args.pred_feat_dim,
            hidden_feat_dim=args.hidden_feat_dim,
            n_input_steps=args.n_input_steps,
            n_pred_steps=args.n_pred_steps,
            constant_var=bool(args.constant_var),
            rnn_cell_type=args.cell,
            name=model_name,
            device=device).to(device)
    
elif args.cell == 'FF':
    model = BBBTimeSeriesPredModel_FF(
            num_hidden_layers=args.nlayers,
            pi=args.pi,
            std1=std1,
            std2=std2,
            gpu=gpu,
            BBB=bool(args.BBB),
            training=args.training,
            sharpen=bool(args.sharpen),
            dropout=args.dropout,
            likelihood_cost_form=args.likelihood_cost_form,
            input_feat_dim=args.input_feat_dim,
            pred_feat_dim=args.pred_feat_dim,
            hidden_feat_dim=args.hidden_feat_dim,
            n_input_steps=args.n_input_steps,
            n_pred_steps=args.n_pred_steps,
            constant_var=bool(args.constant_var),
            name=model_name,
            device=device).to(device)

ut.load_final_model_by_name(model)
model.eval()

# read training set 
training_set = data_ut.read_highd_data(
    'highd_processed_tracks01-60_fr05_loc123456_p0.30', 
    args.batch_size, device)

# pick arbitrary test with some of trajectories
np.random.seed(0)
n_batches = len(training_set)
split = 0.05
ind = np.random.choice(range(n_batches), size=(int(n_batches * split),), replace=False)
test_set_batches = [training_set[i] for i in ind]

rwses = []
rmses = []
for test_set in tqdm(test_set_batches):
    # calculate metrics and return results
    rwses.append(rwse(model, test_set, n_samples=100).detach().item())
    rmses.append(rmse(model, test_set, n_samples=100).detach().item())
print("RWSE: {:.3f} +/ {:.3f}".format(np.array(rwses).mean(), np.array(rwses).std()))
print("RMSE: {:.3f} +/ {:.3f}".format(np.array(rmses).mean(), np.array(rmses).std()))

