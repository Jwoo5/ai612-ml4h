from os import posix_fadvise
import numpy as np
import pickle

with open('../20218078/X_test.pkl', 'rb') as f:
    data = pickle.load(f)

means = []
stds = []
pos = np.zeros(14)
for sample in data:
    means.append(np.mean(sample['img']))
    stds.append(np.std(sample['img']))
    pos += sample['labels']

mean = np.mean(means)
std = np.mean(stds)

print(f"mean: {mean:.3f}")
print(f"std: {std:.3f}")
print(" - class weights:")
print(len(data))
for n_pos in pos:
    print(f"{n_pos}, {(len(data) - n_pos) / n_pos :.3f}")