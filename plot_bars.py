import matplotlib.pyplot as plt
import numpy as np
import argparse

# RWSE
# FF: 0.304 +/ 0.046
# LSTM: 0.151 +/ 0.032
# LSTM + BBB: 0.113 +/ 0.038
# LSTM + BBB + Sharpen: 

# MSE 
# FF: 0.191 +/ 0.040
# LSTM: 0.103 +/ 0.035
# LSTM + BBB: 0.111 +/ 0.038
# LSTM + BBB + sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--objective', '-o', type=str, default='rwse')
args = parser.parse_args()


algos = ('FF', 'LSTM', 'LSTM-BBB', 'LSTM-BBB-Sharpen')
x = np.arange(len(algos))

rwses = [0.304, 0.151, 0.113, 0.1]
rwses_yerr = [0.046, 0.032, 0.038, 0.03]

rmses = [0.191, 0.103, 0.111, 0.1]
rmses_yerr = [0.04, 0.035, 0.038, 0.04]

if args.objective == 'rwse':
    height = rwses
    yerr = rwses_yerr
elif args.objective == 'rmse':
    height = rmses
    yerr = rmses_yerr

plt.bar(x, height, yerr=yerr, capsize=3)
plt.xticks(x, algos)
plt.ylabel(args.objective)
plt.show()
