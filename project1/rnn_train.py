import pickle
import numpy as np
from sklearn.metrics.ranking import precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

train_raw_x = np.load('./20218078/X_train_rnn.npy')
train_y = np.load('./20218078/y_train.npy')

test_raw_x = np.load('./20218078/X_test_rnn.npy')
test_y = np.load('./20218078/y_test.npy')

train_raw_x = [x.split(' ') for x in train_raw_x]
test_raw_x = [x.split(' ') for x in test_raw_x]

train_x_ = []
test_x_ = []

train_x = []
test_x =  []

codebook = {
    'UNK' : 0,
    'PAD' : 1
}

for x in train_raw_x:
    x = [k.split(':')[1]  for k in x]
    train_x_.append(x)

for x in test_raw_x:
    x = [k.split(':')[1] for k in x]
    test_x_.append(x)

for x in train_x_:
    for code in x:
        if not code in codebook:
            codebook[code] = len(codebook)

for x in train_x_:
    x = [codebook[code] for code in x]
    train_x.append(x)

for x in test_x_:
    x = [codebook[code] if code in codebook else codebook['UNK'] for code in x]
    test_x.append(x)

del train_raw_x, test_raw_x, train_x_, test_x_

class MyDataset(Dataset):
    def __init__(self, dataset, label, max_seq = 100):
        dataset = np.array([np.array(data + ([codebook['PAD']] * (max_seq - len(data)))) if len(data) < max_seq \
                            else np.array(data[:max_seq]) for data in dataset])
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.LongTensor(self.dataset[index]), torch.LongTensor([self.label[index]])

class RNN_MP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = kwargs['embed_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.num_layers = kwargs['num_layers']
        self.n_class = kwargs['n_class']

        self.embedding = nn.Embedding(len(codebook), self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, self.num_layers)
        self.proj = nn.Linear(self.hidden_dim, self.n_class)

    def forward(self, source):
        x = self.embedding(source).transpose(0,1)
        output, x = self.gru(x)
        output = torch.mean(output, dim = 0)
        x = self.proj(output)

        return x

batch_size = 512
train_dataset = MyDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

gpu = 0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

embed_dim = 128
hidden_dim = 128
num_layers = 1
n_class = 2
model = RNN_MP(embed_dim = embed_dim, hidden_dim = hidden_dim, num_layers = num_layers, n_class = n_class)

lr = 5e-3
pos = np.sum(train_y)
neg = len(train_y) - pos
weights = torch.FloatTensor([neg, pos])
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
weights = weights.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = [0.9, 0.98], eps = 1e-6, weight_decay = 0.01)
criterion = torch.nn.CrossEntropyLoss(weight = weights)

model.to(device)

epochs = 100
with trange(1, epochs + 1) as pbar:
    for epoch in trange(epochs, ):
        model.train()
        total_loss = 0
        loss_mean = 0
        probs = []
        for step, batch in enumerate(train_loader):
            input, label = batch
            input = input.to(device)
            label = label.squeeze(1).to(device)

            optimizer.zero_grad()

            with torch.autograd.profiler.record_function("forward"):
                output = model(input)
                prob = torch.nn.functional.softmax(output.data, dim = -1).cpu()
                probs.append(prob)
                loss = criterion(output, label)
            
            with torch.autograd.profiler.record_function("backward"):
                loss.backward()
                optimizer.step()
            
            total_loss += loss

        probs = np.concatenate(probs)
        train_AUROC = roc_auc_score(train_y, probs[:,1])
        train_AUPRC = average_precision_score(train_y, probs[:,1])

        loss_mean = total_loss / step

        pbar.set_postfix_str(f"loss: {loss_mean:.3f}, auroc: {train_AUROC:.3f}, auprc: {train_AUPRC:.3f}")

torch.save({
    "state_dict" : model.state_dict()
}, './20218078/rnn_trained.pth')