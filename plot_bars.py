import matplotlib.pyplot as plt
import numpy as np

# RWSE
# FF: 0.2 +/- 0.06
# LSTM: 0.16276285 +/- 0.035808478
# LSTM + BBB: 0.116339006 +/- 0.0412607
# LSTM + BBB + Sharpen: 0.10 +/- 0.04

# MSE 
# FF: 0.14
# LSTM: 6.111288 +/- 1.9257331
# LSTM + BBB: 0.012095822 +/- 0.008749834

objects = ('FF', 'LSTM', 'LSTM-BBB', 'LSTM-BBB-Sharpen')
x = np.arange(len(objects))
height = [0.2, 0.16276285, 0.116339006, 0.1]
yerr = [0.06, 0.0358, 0.04, 0.04]

plt.bar(x, height, yerr=yerr, capsize=3)
plt.xticks(x, objects)
plt.ylabel('RWSE')
plt.show()
