import pickle
from tqdm import tqdm

print('[Ready] prepare data')
vocab = {'UNK': 0,
         'PAD': 1}
counts = {}
threshold = 2

with open('train_tokenized_soft.pkl', 'rb') as f:
    train_notes = pickle.load(f)

print('[Process] fetch samples')
with tqdm(total = len(train_notes)) as pbar:
    for _, sample in train_notes.items():
        for word in sample:
            try:
                counts[word] += 1
            except:
                counts[word] = 1
        pbar.update(1)

print('[Process] extract vocabs for words')
cnt = 2
with tqdm(total = len(counts)) as pbar:
    for word in counts:
        if counts[word] >= 5:
            vocab[word] = cnt
            cnt += 1
        pbar.update(1)

print('[End] dump the vocab file')
with open('train.vocab', mode = 'wb') as fp:
    pickle.dump(vocab, fp)

print(len(vocab))