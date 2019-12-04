class BBBTimeSeriesPredModel_FF(nn.Module):

    def __init__(self, 
        input_feat_dim,
        pred_feat_dim,
        hidden_feat_dim,
        n_input_steps,
        n_pred_steps,
        device,
        num_rnn_layers=1,
        dropout=0,
        BBB=True,
        sharpen=False,
        training=True,
        likelihood_cost_form='gaussian',
        constant_var=False,
        name='model',
        *args, **kwargs):

        super(BBBTimeSeriesPredModel, self).__init__()
        # input size = [batch_size, input_size]
        # input_size = input_seq_len * input_feat_dim (50 * 4)
        # output_size = output_seq_len * output_feat_dim + var_dim (20 * 2 + 1)
        self.net = nn.Sequential(
                    nn.BBBLinear(input_size, 500),
                    nn.ELU(),
                    nn.BBBLinear(500, 500),
                    nn.ELU(),
                    nn.BBBLinear(500, output_size),
                )