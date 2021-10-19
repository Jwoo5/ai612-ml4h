import pandas as pd
import numpy as np
import datetime

file1 = '../mimiciii/ICUSTAYS.csv'
file2 = '../mimiciii/ADMISSIONS.csv'

icu_stays = pd.read_csv(file1)
admissions = pd.read_csv(file2)

train = []
test = []

is_train = False

def dead(start, inter, end):
    def str_to_datetime(string):
        string = string.split(' ')
        string = list(np.array([string[0].split('-'), string[1].split(':')]).flatten())
        string = list(map(int,string))
        return datetime.datetime(string[0], string[1], string[2], string[3], string[4], string[5])
    
    start = str_to_datetime(start)
    inter = str_to_datetime(inter)
    end = str_to_datetime(end)

    if start <= inter and inter <= end:
        return True
    else:
        return False

for hadm_id, icu_id, intime, outtime, los in zip(icu_stays['HADM_ID'],icu_stays['ICUSTAY_ID'], icu_stays['INTIME'], icu_stays['OUTTIME'], icu_stays['LOS']):
    if los < 1 or los > 2:
        continue

    if isinstance(outtime, float):
        continue

    if int(str(icu_id)[-1]) >= 8:
        is_train = False
    else:
        is_train = True

    deathtime = admissions.loc[admissions['HADM_ID'] == hadm_id]['DEATHTIME'].to_numpy()[0]

    if isinstance(deathtime, float):
        # train.append([icu_id, 0]) if is_train else test.append([icu_id, 0])
        train.append(0) if is_train else test.append(0)
        continue
    else:
        if dead(intime, deathtime, outtime):
            # train.append([icu_id, 1]) if is_train else test.append([icu_id, 1])
            train.append(1) if is_train else test.append(1)
            # print(intime, deathtime, outtime)
            a.append(icu_id)
            b.append(intime)
            c.append(deathtime)
            d.append(outtime)
        else:
            # train.append([icu_id, 0]) if is_train else test.append([icu_id, 0])
            train.append(0) if is_train else test.append(0)
    

train = np.array(train)
test = np.array(test)
# if debug:
# np.save('./y_train_debug', train)
# np.save('./y_test_debug', test)

# np.save('y_train', train)
# np.save('y_test', test)

print('DONE!')

print("Statistics - (#0, #1)")
print(f"Train: {np.sum(np.where(train == 0, True, False))} , {np.sum(np.where(train == 1, True, False))}")
print(f"Test: {np.sum(np.where(test == 0, True, False))} , {np.sum(np.where(test == 1, True, False))}")