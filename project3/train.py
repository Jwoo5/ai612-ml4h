import numpy as np
from sklearn.metrics.ranking import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import pickle

class CXRDataset(Dataset):
    def __init__(self, path, transform = None):
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]['img']
        labels = self.dataset[index]['labels']

        if self.transform:
            img = self.transform(img)

        return {
            'inputs': img,
            'labels': labels
        }

class StackChannel(object):    
    def __call__(self, sample):
        sample = np.stack((sample,)*3, axis = 0)
        return sample.transpose((1,2,0))

transform = transforms.Compose([
    StackChannel(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[119.635, 119.635, 119.635], std = [76.647, 76.647, 76.647])
])

batch_size = 64
lr = 5e-5
n_epochs = 3
log_interval_step = 50

train_dataset = CXRDataset('X_train.pkl', transform = transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

valid_dataset = CXRDataset('X_test.pkl', transform = transform)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size * 2)

# pos_weights = torch.tensor([3.631, 4.507, 24.751, 7.704, 27.488, 46.955, 37.627, 3.430, 1.974, 4.322, 175.530, 14.919, 31.932, 2.704]).to('cuda')

criterion = nn.BCEWithLogitsLoss()

model = resnet18(pretrained = True)
model.fc = nn.Linear(512, 14)
model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def train(epoch):
    model.train()
    total_loss = 0
    preds = []
    tgts = []
    for i, batch in enumerate(train_loader):
        inputs = batch['inputs'].to('cuda')
        labels = batch['labels'].to('cuda')

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        preds.append(probs)
        tgts.append(labels)

        if (i+1) % log_interval_step == 0:
            print(f" - [Train] Avg. loss per last {log_interval_step} batches: {total_loss / log_interval_step:.4f}")
            print(f" - [Train] Epoch: {epoch} Step: {i+1}/{len(train_loader)}, loss={loss:.4f}")
            total_loss = 0

def validate():
    model.eval()
    total_loss = 0
    preds = []
    tgts = []
    with tqdm(total = len(valid_loader), desc = f" - [Test] ") as pbar:
        for i, batch in enumerate(valid_loader):
            inputs = batch['inputs'].to('cuda')
            labels = batch['labels'].to('cuda')

            with torch.no_grad():
                outputs = model(inputs)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            preds.append(probs)
            tgts.append(labels)

            pbar.update(1)
    
    preds = np.concatenate(preds, axis = 0)
    tgts = np.concatenate(tgts, axis = 0)

    survive = np.where(tgts == 0, True, False).all(axis = 0)
    survive = np.where(survive == False)[0]

    print(" - [Test] Validation starts")
    auroc_micro = roc_auc_score(tgts, preds, average = 'micro')
    auprc_micro = average_precision_score(tgts, preds, average = 'micro')

    preds = preds[:, survive]
    tgts = tgts[:, survive]

    auroc_macro = roc_auc_score(tgts, preds, average = 'macro')
    auprc_macro = average_precision_score(tgts, preds, average = 'macro')

    print(f" - [Test] auroc_macro: {auroc_macro:.4f}")
    print(f" - [Test] auroc_micro: {auroc_micro:.4f}")
    print(f" - [Test] auprc_macro: {auprc_macro:.4f}")
    print(f" - [Test] auprc_micro: {auprc_micro:.4f}")

for epoch in range(n_epochs):
    train(epoch)
    validate()

torch.save(model.state_dict(), 'trained_model.pt')