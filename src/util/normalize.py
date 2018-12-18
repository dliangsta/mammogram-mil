from load_data import load_data
import numpy as np

x_train, y_train = load_data('train', '../../data/shuffle_within_case', (299,299))
mean, std = np.mean(x_train), np.std(x_train)

print(mean, std)
with open('../../data/metadata.txt','w') as f:
  f.write("{},{}".format(mean,std))