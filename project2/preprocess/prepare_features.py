import pickle
import numpy as np

from tqdm import tqdm

print("[Ready] prepare text data and vocab")

with open('train.vocab', 'rb') as f:
    vocab = pickle.load(f)

with open('train_tokenized_soft.pkl', 'rb') as f:
    train_notes = pickle.load(f)

with open('test_tokenized_soft.pkl', 'rb') as f:
    test_notes = pickle.load(f)

_train_notes = []
print("[Train] transform word into index in vocab")
with tqdm(total = len(train_notes)) as pbar:
    for hadm in train_notes:
        _train_notes.append(np.array(list(map(lambda x: vocab[x] if x in vocab else vocab['UNK'], train_notes[hadm][:5000]))))
        pbar.update(1)
_train_notes = np.array(_train_notes)

_test_notes = []
print("[Test] transform word into index in vocab")
with tqdm(total = len(test_notes)) as pbar:
    for hadm in test_notes:
        _test_notes.append(np.array(list(map(lambda x: vocab[x] if x in vocab else vocab['UNK'], test_notes[hadm][:5000]))))
        pbar.update(1)
_test_notes = np.array(_test_notes)

print("[End] dump the feature files")
np.save("../20218078/X_train", _train_notes)
np.save("../20218078/X_test", _test_notes)
