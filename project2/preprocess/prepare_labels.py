import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("[Ready] prepare data files")

with open('train_tokenized_soft.pkl', 'rb') as f:
    train_notes = pickle.load(f)

with open('test_tokenized_soft.pkl', 'rb') as f:
    test_notes = pickle.load(f)

df_px = pd.read_csv('../../mimiciii/PROCEDURES_ICD.csv', dtype = {'ICD9_CODE': object})
df_dx = pd.read_csv('../../mimiciii/DIAGNOSES_ICD.csv', dtype = {'ICD9_CODE': object})

df_px_codes = pd.read_csv('../../mimiciii/D_ICD_PROCEDURES.csv', dtype = {'ICD9_CODE': object})
df_dx_codes = pd.read_csv('../../mimiciii/D_ICD_DIAGNOSES.csv', dtype = {'ICD9_CODE': object})

train_labels = {}
test_labels = {}

print("[Ready] prepare frame for multi hot encoding")
px_codes = (df_px_codes['ICD9_CODE'].apply(lambda x: 'P_'+str(x))).to_list()
dx_codes = (df_dx_codes['ICD9_CODE'].apply(lambda x: 'D_'+str(x))).to_list()

mlb = MultiLabelBinarizer()
mlb.fit([px_codes + dx_codes])

print("[Train] extract labels for training dataset")

with tqdm(total = len(train_notes)) as pbar:
    for hadm in train_notes:
        procedures = set(df_px.loc[df_px['HADM_ID'] == hadm]['ICD9_CODE'].apply(lambda x: 'P_'+str(x)))
        diagnoses = set(df_dx.loc[df_dx['HADM_ID'] == hadm]['ICD9_CODE'].apply(lambda x: 'D_'+str(x)))
        codes = [procedures.union(diagnoses)]

        train_labels[hadm] = mlb.transform(codes)

        pbar.update(1)

print("[Test] extract labels for test dataset")

with tqdm(total = len(test_notes)) as pbar:
    for hadm in test_notes:
        procedures = set(df_px.loc[df_px['HADM_ID'] == hadm]['ICD9_CODE'].apply(lambda x: 'P_'+str(x)))
        diagnoses = set(df_dx.loc[df_dx['HADM_ID'] == hadm]['ICD9_CODE'].apply(lambda x: 'D_'+str(x)))
        codes = [procedures.union(diagnoses)]

        test_labels[hadm] = mlb.transform(codes)

        pbar.update(1)

print("[Final] dump the label files. It can take a long time.")
train_cnt = 0
test_cnt = 0

with open('y_train.txt', 'w') as f:
    with tqdm(total = len(train_labels)) as pbar:
        for hadm in train_labels:
            str1 = str(hadm) + ','
            str2 = ''
            for b_label in train_labels[hadm][0]:
                if b_label == 1:
                    train_cnt += 1
                str2 += str(b_label) + ','
            str2 = str2.rstrip(',')

            f.write(str1 + str2 + '\n')

            pbar.update(1)

with open('y_test.txt', 'w') as f:
    with tqdm(total = len(test_labels)) as pbar:
        for hadm in test_labels:
            str1 = str(hadm) + ','
            str2 = ''
            for b_label in test_labels[hadm][0]:
                if b_label == 1:
                    test_cnt += 1
                str2 += str(b_label) + ','
            str2 = str2.rstrip(',')

            f.write(str1 + str2 + '\n')

            pbar.update(1)

print(f"[End] train_pos: {train_cnt}")  # 655751
print(f"[End] test_pos : {test_cnt}")   # 163584