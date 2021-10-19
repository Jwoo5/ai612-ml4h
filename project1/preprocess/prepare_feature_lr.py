import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb

with open('train_charts_for_each_icu.pickle', 'rb') as f:
    train_charts = pickle.load(f)
with open('test_charts_for_each_icu.pickle', 'rb') as f:
    test_charts = pickle.load(f)
with open('codebook.pickle', 'rb') as f:
    codebook = pickle.load(f)

train_y = np.load('y_train.npy')
test_y = np.load('y_test.npy')

print(len(train_charts))
print(len(train_y))

print(len(test_charts))
print(len(test_y))

train_features = []
test_features = []

train_y_revised = []
test_y_revised = []

train_debug = 0
test_debug = 0

with tqdm(total = len(train_charts) + len(test_charts)) as pbar:
    for icu, label in zip(train_charts, train_y):
        chart_times = np.array(train_charts[icu]['chart_time'])
        items = np.array(train_charts[icu]['chart_event'])
        values = np.array(train_charts[icu]['chart_value'])
        valuenums = np.array(train_charts[icu]['chart_valuenum'])

        if len(chart_times) == 0:
            continue

        index = np.argsort(chart_times)
        chart_times = chart_times[index]
        items = items[index]
        values = values[index]
        valuenums = valuenums[index]

        index = np.where(valuenums != '')
        chart_times = chart_times[index]
        items = items[index]
        values = values[index]
        valuenums = valuenums[index]

        if len(chart_times) == 0:
            continue

        feature = np.zeros(len(codebook))
        for item, value in zip(items, valuenums):
            if item in codebook:
                feature[codebook[item]] = value
        
        train_features.append(feature)
        train_y_revised.append(label)
        train_debug += 1

        pbar.update(1)

    for icu, label in zip(test_charts, test_y):
        chart_times = np.array(test_charts[icu]['chart_time'])
        items = np.array(test_charts[icu]['chart_event'])
        values = np.array(test_charts[icu]['chart_value'])
        valuenums = np.array(test_charts[icu]['chart_valuenum'])

        if len(chart_times) == 0:
            continue

        index = np.argsort(chart_times)
        chart_times = chart_times[index]
        items = items[index]
        values = values[index]
        valuenums = valuenums[index]

        index = np.where(valuenums != '')
        chart_times = chart_times[index]
        items = items[index]
        values = values[index]
        valuenums = valuenums[index]

        if len(chart_times) == 0:
            continue

        feature = np.zeros(len(codebook))
        for item, value in zip(items, valuenums):
            if item in codebook:
                feature[codebook[item]] = value
        
        test_features.append(feature)
        test_y_revised.append(label)
        test_debug += 1

        pbar.update(1)

if train_debug != len(train_y_revised):
    print(f"train_debug: {train_debug}, train-y: {len(train_y_revised)}, train_charts: {len(train_charts)}")
if test_debug != len(test_y_revised):
    print(f"test_debug: {test_debug}, test_y: {len(test_y_revised)}, test_charts: {len(test_charts)}")

train_features = np.array(train_features)
test_features = np.array(test_features)
train_y_revised = np.array(train_y_revised)
test_y_revised = np.array(test_y_revised)

print(len(train_features))
print(len(train_y_revised))
print(len(test_features))
print(len(test_y_revised))

"""normlizing"""
# train_features_norm = train_features.copy()
# test_features_norm = test_features.copy()
# features_norm = np.concatenate((train_features_norm, test_features_norm))

# mean = np.mean(features_norm, axis = 0)
# std = np.std(features_norm, axis = 0)

# features_norm -= mean
# features_norm /= (std + 1e-6)

# train_features_norm = features_norm[:len(train_y)]
# test_features_norm = features_norm[len(train_y):]

# np.save('../X_train_lr_norm', train_features_norm)
# np.save('../X_test_lr_norm', test_features_norm)

np.save('../20218078/X_train_logistic', train_features)
np.save('../20218078/X_test_logistic', test_features)
np.save('../20218078/y_train', train_y_revised)
np.save('../20218078/y_test', test_y_revised)