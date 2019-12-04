import torch
from torchsummary import summary
import argparse
import numpy as np
from codebase.models.BBBTimeSeriesPredModel import BBBTimeSeriesPredModel
from codebase.models.BBBTimeSeriesPredModel_FF import BBBTimeSeriesPredModel_FF
from codebase.train import train
import codebase.utils as ut
import data.data_utils as data_ut


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
parser.add_argument('--constant_var', action='store_true')
parser.add_argument('--BBB', action='store_true')
parser.add_argument('--sharpen', action='store_true')
parser.add_argument('--likelihood_cost_form', type=str, default='gaussian')
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--pi', type=float, default=0.25)
parser.add_argument('--logstd1', type=int, default=-1)
parser.add_argument('--logstd2', type=int, default=-6)
# Train
parser.add_argument('--dev_mode', action='store_true')
parser.add_argument('--training', action='store_true')
parser.add_argument('--clip_grad', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--iter_max', type=int, default=400000)
parser.add_argument('--iter_plot', type=int, default=2000)
parser.add_argument('--iter_save', type=int, default=2000)

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
        ('BBB={:s}', str(args.BBB)),
        ('data={:s}', args.dataset_name),
        ('nlayers={:d}', args.nlayers),
        ('nhid={:d}', args.hidden_feat_dim),
        ('const_var={}', args.constant_var),
        ('dropout={:.1f}', args.dropout),
        ('clipgrad={}', str(args.clip_grad)),
        ('loss={:s}', args.likelihood_cost_form),
        ('run={:d}', args.run),
    ]
else:
    layout = [
        ('model={:s}', args.cell),
        ('BBB={:s}', str(args.BBB)),
        ('data={:s}', args.dataset_name),
        ('nlayers={:d}', args.nlayers),
        ('nhid={:d}', args.hidden_feat_dim),
        ('const_var={}', args.constant_var),
        ('dropout={:.1f}', args.dropout),
        ('clipgrad={}', str(args.clip_grad)),
        ('loss={:s}', args.likelihood_cost_form),
        ('sharpen={:s}', str(args.sharpen)),
        ('pi={:.2f}', args.pi),
        ('logstd1={:d}', args.logstd1),
        ('logstd2={:d}', args.logstd2),
        ('run={:d}', args.run),
    ]

model_name = '_'.join([t.format(v) for (t, v) in layout])
print(model_name, '\n')

if not args.dev_mode:
    ut.prepare_dirs(model_name, overwrite_existing=True)
else:
    args.iter_plot = np.inf

# dataset_name = 'dummy{}d'.format(input_feat_dim)
# n_batches = 2000 # only used by dummy data
# training_set = data_ut.dummy_data_creator(
#         batch_size=batch_size, 
#         n_batches=n_batches, 
#         input_feat_dim=input_feat_dim,
#         n_input_steps=n_input_steps, 
#         n_pred_steps=n_pred_steps,
#         kernel=data_ut.sinusoidal_kernel,
#         device=device)

training_set = data_ut.read_highd_data(
    'highd_processed_tracks01-60_fr05_loc123456_p0.30', 
    args.batch_size, device)
n_batches = len(training_set)

if args.cell == 'LSTM':
    model = BBBTimeSeriesPredModel(
            num_rnn_layers=args.nlayers,
            pi=args.pi,
            std1=std1,
            std2=std2,
            gpu=gpu,
            BBB=args.BBB,
            training=args.training,
            sharpen=args.sharpen,
            dropout=args.dropout,
            likelihood_cost_form=args.likelihood_cost_form,
            input_feat_dim=args.input_feat_dim,
            pred_feat_dim=args.pred_feat_dim,
            hidden_feat_dim=args.hidden_feat_dim,
            n_input_steps=args.n_input_steps,
            n_pred_steps=args.n_pred_steps,
            constant_var=args.constant_var,
            rnn_cell_type=args.cell,
            name=model_name,
            device=device).to(device)
    
elif args.cell == 'FF'
    model = BBBTimeSeriesPredModel_FF(
            num_hidden_layers=args.nlayers,
            #pi=args.pi,
            #std1=std1,
            #std2=std2,
            gpu=gpu,
            BBB=args.BBB,
            training=args.training,
            sharpen=args.sharpen,
            dropout=args.dropout,
            #likelihood_cost_form=args.likelihood_cost_form,
            input_feat_dim=args.input_feat_dim,
            pred_feat_dim=args.pred_feat_dim,
            hidden_feat_dim=args.hidden_feat_dim,
            n_input_steps=args.n_input_steps,
            n_pred_steps=args.n_pred_steps,
            constant_var=args.constant_var,
            #rnn_cell_type=args.cell,
            name=model_name,
            device=device).to(device)
    
train(model, training_set, args.batch_size, n_batches, 
        kernel=data_ut.sinusoidal_kernel,
        lr=args.lr,
        clip_grad=args.clip_grad,
        iter_max=args.iter_max, 
        iter_save=args.iter_save, 
        iter_plot=args.iter_plot, 
        reinitialize=False)

