import os
import pandas as pd
from tqdm import tqdm

path = '/home/data_storage/mimic-cxr/dataset/original_mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/'
file = 'mimic-cxr-2.0.0-metadata.csv'

df = pd.read_csv(os.path.join(path, file))
bucket = []

if os.path.exists('test.tsv'):
    os.remove('test.tsv')
if os.path.exists('train.tsv'):
    os.remove('train.tsv')

with tqdm(total = len(df)) as pbar:
    for study_id, subject_id in zip(df['study_id'], df['subject_id']):
        if subject_id in bucket:
            pbar.update(1)
            continue

        dicom_id = df.loc[(df['study_id'] == study_id) & (df['ViewPosition'] == 'AP')]['dicom_id']
        if len(dicom_id) == 0:
            pbar.update(1)
            continue
        dicom_id = dicom_id.iloc[0]

        manifest = os.path.join('p' + str(subject_id)[:2], 'p' + str(subject_id), 's' + str(study_id), dicom_id + '.jpg')

        if int(str(study_id)[-1]) >= 8:
            with open('test.tsv', 'a') as f:
                f.write(manifest + '\n')
        else:
            with open('train.tsv', 'a') as f:
                f.write(manifest + '\n')

        bucket.append(subject_id)
        pbar.update(1)