import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

path = '/home/data_storage/mimic-cxr/dataset/original_mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

train_lst = []
test_lst = []

def strip(x):
    return x[:-1]

def one_hot_encoding(lst):
    lst = lst.split(',')[1:]
    lst = list(map(lambda x: int(x), lst))
    lst = np.array(lst)
    return lst

with open('train.tsv', 'r') as f:
    train_files = f.readlines()
    train_files = list(map(strip, train_files))

with open('y_train.txt', 'r') as f:
    y_train = f.readlines()
    y_train = list(map(strip, y_train))
    y_train = list(map(one_hot_encoding, y_train))

with open('test.tsv', 'r') as f:
    test_files = f.readlines()
    test_files = list(map(strip, test_files))

with open('y_test.txt', 'r') as f:
    y_test = f.readlines()
    y_test = list(map(strip, y_test))
    y_test = list(map(one_hot_encoding, y_test))


with tqdm(total = len(train_files)) as pbar:
    for i, train_file in enumerate(train_files):
        img = Image.open(os.path.join(path,train_file))
        img = img.resize((256,256))
        img = np.array(img)
        labels = y_train[i]

        train_lst.append({
            'img': img,
            'labels': labels
        })

        pbar.update(1)
train_lst = np.stack(train_lst, axis = 0)

with tqdm(total = len(test_files)) as pbar:
    for i, test_file in enumerate(test_files):
        img = Image.open(os.path.join(path, test_file))
        img = img.resize((256,256))
        img = np.array(img)
        labels = y_test[i]

        test_lst.append({
            'img': img,
            'labels': labels
        })

        pbar.update(1)
test_lst = np.stack(test_lst, axis = 0)

with open('X_train.pkl', 'wb') as f:
    pickle.dump(train_lst, f)

with open('X_test.pkl', 'wb') as f:
    pickle.dump(test_lst, f)