import os
import csv

path = '/home/data_storage/mimic-cxr/dataset/original_mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/'
file = 'mimic-cxr-2.0.0-negbio.csv'

if os.path.exists('y_train.txt'):
    os.remove('y_train.txt')
if os.path.exists('y_test.txt'):
    os.remove('y_test.txt')

def convert_to_labels(x):
    if x == '1.0':
        return '1'
    else:
        return '0'

with open('train.tsv', 'r') as f:
    trains = f.readlines()
    trains = list(map(lambda x: x.split('/')[2][1:], trains))
with open('test.tsv', 'r') as f:
    tests = f.readlines()
    tests = list(map(lambda x: x.split('/')[2][1:], tests))

with open(os.path.join(path, file), 'r') as f:
    reader = csv.reader(f)
    next(iter(reader))

    for row in reader:
        study_id = row[1]
        row = list(map(convert_to_labels, row[2:]))
        row = ('s' + study_id +',') + ','.join(row)


        if study_id in trains:
            with open('y_train.txt', 'a') as fp:
                fp.write(row + '\n')
        elif study_id in tests:
            with open('y_test.txt', 'a') as fp:
                fp.write(row + '\n')