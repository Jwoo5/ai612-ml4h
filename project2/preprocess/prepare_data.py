import pandas as pd
import csv
import pickle
import numpy as np
import datetime
from datetime import timedelta
from tqdm import tqdm
import pdb

fn_adms = '../../mimiciii/ADMISSIONS.csv'
fn_note = '../../mimiciii/NOTEEVENTS.csv'

df_adms = pd.read_csv(fn_adms)

train = {}
test = {}

print("[Stage 1] Prepare data split")

for hadm_id in df_adms['HADM_ID']:
    if isinstance(hadm_id, float):
        continue
    
    if int(str(hadm_id)[-1]) >= 8:
        test[hadm_id] = ''
    else:
        train[hadm_id] = ''

print("[Stage 1] DONE")

print("[Stage 2] Join note events with valid HADM_ID")
with open(fn_note, mode = 'r') as noteevents:
    noteevents = csv.reader(noteevents)
    header = next(iter(noteevents))

    code = {col : code for code, col in enumerate(header)}

    with tqdm(total = 2083180, desc = "Processing: ") as pbar:
        for i, event in enumerate(noteevents):
            if event[code['CATEGORY']] != 'Discharge summary':
                pbar.update(1)
                continue
            if event[code['DESCRIPTION']] != 'Report':
                pbar.update(1)
                continue
            
            hadm_id = int(event[code['HADM_ID']])
            text = event[code['TEXT']]


            if hadm_id in train:
                train[hadm_id] = text
            elif hadm_id in test:
                test[hadm_id] = text
            pbar.update(1)

with open('train_notes_for_each_hadm.pkl', mode = 'wb') as fp:
    pickle.dump(train, fp)

with open('test_notes_for_each_hadm.pkl', mode = 'wb') as fp:
    pickle.dump(test, fp)