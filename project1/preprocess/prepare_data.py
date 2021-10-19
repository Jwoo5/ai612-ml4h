import pandas as pd
import csv
import pickle
import numpy as np
import datetime
from datetime import timedelta
from tqdm import tqdm
import pdb

file1 = '../mimiciii/ICUSTAYS.csv'
file2 = '../mimiciii/CHARTEVENTS.csv'

def str_to_datetime(string):
    string = string.split(' ')
    string = list(np.array([string[0].split('-'), string[1].split(':')]).flatten())
    string = list(map(int,string))
    return datetime.datetime(string[0], string[1], string[2], string[3], string[4], string[5])

icu_stays = pd.read_csv(file1)
train = {}
test = {}

print("Load valid ICUSTAY_IDs")

for icu_id, intime, outtime, los in zip(icu_stays['ICUSTAY_ID'], icu_stays['INTIME'], icu_stays['OUTTIME'], icu_stays['LOS']):
    if los < 1 or los > 2:
        continue

    if isinstance(outtime, float):
        continue

    if int(str(icu_id)[-1]) >= 8:
        test[icu_id] = {}
        test[icu_id]['in_time'] = str_to_datetime(intime)
        test[icu_id]['chart_event'] = []
        test[icu_id]['chart_value'] = []
        test[icu_id]['chart_valuenum'] = []
        test[icu_id]['chart_time'] = []
    else:
        train[icu_id] = {}
        train[icu_id]['in_time'] = str_to_datetime(intime)
        train[icu_id]['chart_event'] = []
        train[icu_id]['chart_value'] = []
        train[icu_id]['chart_valuenum'] = []
        train[icu_id]['chart_time'] = []
print("DONE!")

prev = None
threshold = 100
thresh_time = timedelta(hours = 3)

print("Join chart events with valid ICUSTAY_ID")
with open(file2, mode = 'r') as chartevents:
    chartevents = csv.reader(chartevents)
    header = next(iter(chartevents))

    code = {col : code for code, col in enumerate(header)}

    with tqdm(total = 330712483, desc = "Processing: ") as pbar:
        for i, event in enumerate(chartevents):
            if event[code['ICUSTAY_ID']] == '':
                pbar.update(1)
                continue

            icu_id = int(event[code['ICUSTAY_ID']])
            charttime = event[code['CHARTTIME']]
            charttime = str_to_datetime(charttime)

            if icu_id in train:
                intime = train[icu_id]['in_time']
                if timedelta() <= charttime - intime and charttime - intime <= thresh_time:
                    train[icu_id]['chart_event'].append(event[code['ITEMID']])
                    train[icu_id]['chart_value'].append(event[code['VALUE']])
                    train[icu_id]['chart_valuenum'].append(event[code['VALUENUM']])
                    train[icu_id]['chart_time'].append(charttime)
                else:
                    pbar.update(1)
                    continue
            elif icu_id in test:
                intime = test[icu_id]['in_time']
                if timedelta() <= charttime - intime and charttime - intime <= thresh_time:
                    test[icu_id]['chart_event'].append(event[code['ITEMID']])
                    test[icu_id]['chart_value'].append(event[code['VALUE']])
                    test[icu_id]['chart_valuenum'].append(event[code['VALUENUM']])
                    test[icu_id]['chart_time'].append(charttime)
                else:
                    pbar.update(1)
                    continue
            pbar.update(1)

with open('train_charts_for_each_icu.pickle', mode = 'wb') as fp:
    pickle.dump(train, fp)

with open('test_charts_for_each_icu.pickle', mode = 'wb') as fp:
    pickle.dump(test, fp)