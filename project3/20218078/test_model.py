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
])

batch_size = 128

valid_dataset = CXRDataset('X_test.pkl', transform = transform)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size * 2)

model = resnet18(pretrained = True)
model.fc = nn.Linear(512, 14)
model.to('cuda')

model.load_state_dict(torch.load('trained_model.pt'))

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

    with open('20218078_model.txt', 'w') as f:
        f.write('20218078\n')
        f.write(f"{auroc_macro:.4f}\n")
        f.write(f"{auroc_micro:.4f}\n")
        f.write(f"{auprc_macro:.4f}\n")
        f.write(f"{auprc_micro:.4f}\n")

validate()