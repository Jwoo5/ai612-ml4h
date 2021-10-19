import re
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt

import copy

with open('train_notes_for_each_hadm.pkl', 'rb') as f:
    train_notes = pickle.load(f)
with open('test_notes_for_each_hadm.pkl', 'rb') as f:
    test_notes = pickle.load(f)

"""
TODO
tokenize를 제대로 하던지, 아니면 그냥 str으로만 잠깐 변환해놓고 나중에 tokenize하던지.
"""

train_lengths = {}
test_lengths = {}

tokenized_train_notes = {}
tokenized_test_notes = {}

print("[Train] preparing training dataset")
with tqdm(total = len(train_notes)) as pbar:
    for hadm in train_notes:
        txt = train_notes[hadm].lower()

        if txt == '':
            pbar.update(1)
            continue
        tokenized_train_notes[hadm] = {}

        txt = txt.replace('.', ' . ')
        txt = txt.replace('(', ' ( ')
        txt = txt.replace(')', ' ) ')
        txt = txt.replace('[', ' [ ')
        txt = txt.replace(']', ' ] ')
        txt = txt.replace('-', ' - ')
        txt = txt.replace('/', ' / ')
        txt = txt.replace('~', ' ~ ')
        txt = txt.replace('>', ' > ')
        txt = txt.replace('<', ' < ')
        txt = re.split(' |\n|\?|\t|\*|#|=|_|xxx+', txt)
        txt = [x for x in txt if x != '']
        tokenized_train_notes[hadm] = txt

        if len(txt) == 0:
            breakpoint()

        try:
            train_lengths[len(txt)] += 1
        except:
            train_lengths[len(txt)] = 1

        pbar.update(1)
print(f"[Train] Length: {len(tokenized_train_notes)}")

print()
print("[Test] preparing test dataset")
with tqdm(total = len(test_notes)) as pbar:
    for hadm in test_notes:
        txt = test_notes[hadm].lower()

        if txt == '':
            pbar.update(1)
            continue
        tokenized_test_notes[hadm] = {}

        txt = txt.replace('.', ' . ')
        txt = txt.replace('(', ' ( ')
        txt = txt.replace(')', ' ) ')
        txt = txt.replace('[', ' [ ')
        txt = txt.replace(']', ' ] ')
        txt = txt.replace('-', ' - ')
        txt = txt.replace('/', ' / ')
        txt = txt.replace('~', ' ~ ')
        txt = txt.replace('>', ' > ')
        txt = txt.replace('<', ' < ')
        txt = re.split(' |\n|\?|\t|\*|#|=|_|xxx+', txt)
        txt = [x for x in txt if x != '']
        tokenized_test_notes[hadm] = txt

        if len(txt) == 0:
            breakpoint()

        try:
            test_lengths[len(txt)] += 1
        except:
            test_lengths[len(txt)] = 1

        pbar.update(1)

print(f"[Test] Length: {len(tokenized_test_notes)}")

x = []
y = []
for length in sorted(train_lengths.items()):
    x.append(length[0])
    y.append(length[1])

plt.clf()
plt.plot(x,y)
plt.savefig('train_lengths.png')

x = []
y = []
for length in sorted(test_lengths.items()):
    x.append(length[0])
    y.append(length[1])

plt.clf()
plt.plot(x,y)
plt.savefig('test_lengths.png')


print(f"[End] dump the tokenized dataset")
with open('train_tokenized_soft.pkl', mode = 'wb') as fp:
    pickle.dump(tokenized_train_notes, fp)

with open('test_tokenized_soft.pkl', mode = 'wb') as fp:
    pickle.dump(tokenized_test_notes, fp)