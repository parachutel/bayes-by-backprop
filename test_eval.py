import torch
from torchsummary import summary
import argparse
import numpy as np
import random
from codebase.models.BBBTimeSeriesPredModel import BBBTimeSeriesPredModel
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
parser.add_argument('--constant_var', type=int, default=0)
parser.add_argument('--BBB', type=int, default=1)
parser.add_argument('--sharpen', type=int, default=0)
parser.add_argument('--likelihood_cost_form', type=str, default='gaussian')
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--pi', type=float, default=0.25)
parser.add_argument('--logstd1', type=int, default=-1)
parser.add_argument('--logstd2', type=int, default=-6)
# Train
parser.add_argument('--clip_grad', type=int, default=5)
parser.add_argument('--run', type=int, default=0)

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

model = BBBTimeSeriesPredModel(
        num_rnn_layers=args.nlayers,
        pi=args.pi,
        std1=std1,
        std2=std2,
        gpu=gpu,
        BBB=bool(args.BBB),
        training=True,
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

ut.load_final_model_by_name(model)
model.eval()


def get_rwse(model, full_true_trajs, n_samples=100):
    """ root-weighted square error (RWSE) captures 
        the deviation of a model’s probability
        mass from real-world trajectories
    """
    n_seqs = full_true_trajs.shape[1]
    inputs = full_true_trajs[:model.n_input_steps, :, :].detach()
    targets = full_true_trajs[model.n_input_steps:, :, :2].detach()

    if model.BBB:
        for i in range(n_samples):
            # not using sharpening
            pred = model.forward(inputs)
            pred = pred.detach()
            if not model.constant_var:
                pred = pred[:, :, :-1]
            mean_sq_err = ((targets - pred) ** 2).sum() / n_seqs

            if i == 0:
                mean_sq_err_list = mean_sq_err.unsqueeze(-1)
            else:
                mean_sq_err = mean_sq_err.unsqueeze(-1)
                mean_sq_err_list = torch.cat((mean_sq_err_list, mean_sq_err), dim=-1)

    else:
        pred = model.forward(inputs)
        pred = pred.detach()
        if not model.constant_var:
            mean, var = ut.gaussian_parameters(pred, dim=-1)
        else:
            mean = pred
            var = model.pred_var

        for i in range(n_samples):
            sample_trajs = ut.sample_gaussian(mean, var)
            mean_sq_err = ((targets - sample_trajs) ** 2).sum() / n_seqs

            if i == 0:
                mean_sq_err_list = mean_sq_err.unsqueeze(-1)
            else:
                mean_sq_err = mean_sq_err.unsqueeze(-1)
                mean_sq_err_list = torch.cat((mean_sq_err_list, mean_sq_err), dim=-1)

    mean_rwse = mean_sq_err_list.mean().sqrt()

    return mean_rwse

def get_mse(model, full_true_trajs, n_samples=100):
    """ root-weighted square error (RWSE) captures 
        the deviation of a model’s probability
        mass from real-world trajectories
    """
    n_seqs = full_true_trajs.shape[1]
    inputs = full_true_trajs[:model.n_input_steps, :, :].detach()
    targets = full_true_trajs[model.n_input_steps:, :, :2].detach()

    if model.BBB:
        for i in range(n_samples):
            # not using sharpening
            pred = model.forward(inputs) # one output sample
            pred = pred.detach()
            if i == 0:
                pred_list = pred.unsqueeze(-1)
            else:
                pred = pred.unsqueeze(-1)
                pred_list = torch.cat((pred_list, pred), dim=-1)

        if model.constant_var:
            mean_pred = pred_list.mean(dim=-1)
            std_pred = pred_list.std(dim=-1)
        else:
            mean_pred = pred_list[:, :, :-1, :].mean(dim=-1)
            std_pred = pred_list[:, :, :-1, :].std(dim=-1)

    else:
        pred = model.forward(inputs)
        pred = pred.detach()
        if not model.constant_var:
            mean, var = ut.gaussian_parameters(pred, dim=-1)
        else:
            mean = pred
            var = model.pred_var

        for i in range(n_samples):
            sample_trajs = ut.sample_gaussian(mean, var)
            sample_trajs = mean

            if i == 0:
                pred_list = sample_trajs.unsqueeze(-1)
            else:
                sample_trajs = sample_trajs.unsqueeze(-1)
                pred_list = torch.cat((pred_list, sample_trajs), dim=-1)

        if model.constant_var:
            mean_pred = pred_list.mean(dim=-1)
            std_pred = pred_list.std(dim=-1)
        else:
            mean_pred = pred_list[:, :, :-1, :].mean(dim=-1)
            std_pred = pred_list[:, :, :-1, :].std(dim=-1)

    mse = ((mean_pred - targets) ** 2).sum() / n_seqs

    return mse




if __name__ == '__main__':

    training_set = data_ut.read_highd_data(
        'highd_processed_tracks01-60_fr05_loc123456_p0.30', 
        args.batch_size, device)
    n_batches = len(training_set)
    
    for i in range(30):
        batch_id = random.sample(range(n_batches), 1)[0]
        mean_rwse = get_rwse(model, training_set[batch_id], n_samples=100)
        mse = get_mse(model, training_set[batch_id], n_samples=100)
        if i == 0:
            mse_list = mse.unsqueeze(-1)
            mean_rwse_list = mean_rwse.unsqueeze(-1)
        else:
            mse = mse.unsqueeze(-1)
            mse_list = torch.cat((mse_list, mse), dim=-1)
            mean_rwse = mean_rwse.unsqueeze(-1)
            mean_rwse_list = torch.cat((mean_rwse_list, mean_rwse), dim=-1)
    
    print('rwse', mean_rwse_list.detach().mean().numpy(), 
            '+/-', mean_rwse_list.detach().std().numpy())
    print('mse', mse_list.detach().mean().numpy(), 
            '+/-', mse_list.detach().std().numpy())

