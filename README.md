# Dev Logs

11/07/19
- init
- import BBBLayers from Yuchen Lu's code
- TODO: figure out the implementation of functions
- TODO: figure out how data are taken in
- TODO: figure out needs for data processing

11/08/19
- figured out working flow of asynchronous many-to-many prediction
- figured out basic network structure
- figured out requirements for time-series data
- tested LSTM with dummy 1d sinusoidal data, worked well
- testing LSTM with dummy 2d sinusoidal data, wait and see
- TODO: test dummy sinusoidal data on BBBRNN

11/09/19
- implemented some data utility functions
- processed stocks data
- ran training on stocks data, not so good for now 
- TODO: tune stocks prediction training


11/11/19
- figured out structure of BBBRNN
- implemented time-series prediction model with BBB
- tested the model, training was intractable
- TODO: loss re-weighting with minibatch
- TODO: inspect details in the model
- TODO: debug training

11/12/19
- training of BBBLSTM can run smoothly
- PROBLEM: BBBLSTM always output 'straight lines', without much expressiveness
- PROBLEM: easily slow down
- PROBLEM: graph reset after each `loss.backward()`, had to use `retain_graph=True`
- TODO: inspect details in the model thoroughly
- TODO: implement evaluation methods (using MC sampling and using mean weights)

11/13/19
- problem located in loss re-weighting
- TODO: @arec extract data and save them as one torch.Tensor with `shape = (full_seq_len, n_tot_sequences, feat_dim)`. Save the tensor using `torch.save(data_tensor, 'data_tensor_name.pt')` (to /data/processed).
- IDEA: using shared variance for each pred feat dim
- IDEA: using fixed variance for decoder output?

11/14/19
- outputting means only, data prob is computed using a constant variance (0.1, as homework 2 does in FSVAE)
- BBBLSTM was able to learn sinusoidal data pretty good
- highD data are processed