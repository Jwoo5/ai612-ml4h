import pickle
import numpy as np
from tqdm import tqdm
import pdb

with open('train_charts_for_each_icu.pickle', 'rb') as f:
    train_charts = pickle.load(f)
with open('test_charts_for_each_icu.pickle', 'rb') as f:
    test_charts = pickle.load(f)

# TODO
# select * from D_ITEMS where param_type = Numeric
# makes sense but has a problem
# there are duplicated item_ids for one concept
# e.g. 211 (CareVue) and 220045 (Metavision) -> Heart Rate
# therefore, we have to consider 211 and 220045 are the same.
# for instance, if we featurize each icu_stay as a vector,
# we can deal with an element of the vector to represent someone's heart rate,
# and we should care about two ids(211, 220045) to fill this element.

# Plus, we have to consider LINKSTO column
# which means that where an itemid is contained (e.g. CHARTEVENTS, ...)
# we should filter out the items for only considering CHARTEVENTS

train_y = np.load('../20218078/y_train.npy')
test_y = np.load('../20218078/y_test.npy')

print(len(train_charts))
print(len(train_y))

print(len(test_charts))
print(len(test_y))

codebook = {}

with tqdm(total = len(train_charts)) as pbar:
    for icu in train_charts:
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

        for item, value in zip(items, valuenums):
            if not item in codebook:
                codebook[item] = len(codebook)
        
        pbar.update(1)

with open('codebook.pickle', mode = 'wb') as fp:
    pickle.dump(codebook, fp)