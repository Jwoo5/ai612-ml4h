import numpy as np

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(np.count_nonzero(y_train == 0))
print(np.count_nonzero(y_train == 1))
print(np.count_nonzero(y_test == 0))
print(np.count_nonzero(y_test == 1))