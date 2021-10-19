import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import math
import copy
from typing import Tuple, List
from tqdm import tqdm

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, padding_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, padding_mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)

        return self.dropout(x)

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx = None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
    
    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2,-1)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        )
        return output.type_as(input)

class ConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm = False,
            is_group_norm = False,
            conv_bias = False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride = stride, bias = conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm ar exclusive"

            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm = mode == "layer_norm",
                    is_group_norm = mode == "default" and i == 0,
                    conv_bias = conv_bias,
                )
            )
            in_d = dim
    
    def forward(self, x):
        # B x T -> B x C x T
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

class Transformer(nn.Module):
    def __init__(self, encoder, src_embed, d_model):
        super().__init__()
        self.encoder = encoder
        #XXX add conv1d feature extractor (assume that max_length is 5000)
        self.feature_extractor = ConvFeatureExtraction(
            conv_layers = [(d_model,2,2), (d_model,2,2), (d_model,2,2)],
            in_d = d_model,
            dropout = 0.0,
        )
        self.proj = nn.Linear(d_model, 18449)
        self.src_embed = src_embed

    def forward(self, src, padding_mask):
        x = self.encode(src, padding_mask)
        x = x.mean(dim = 1)
        x = self.proj(x)
        return x
    
    def encode(self, src, padding_mask):
        x = self.src_embed(src).transpose(-2,-1)
        x = self.feature_extractor(x).transpose(-2,-1)
        return self.encoder(x, padding_mask = None)

def make_model(src_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model,c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        d_model
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

class MyDataset(Dataset):
    def __init__(self, train = True) -> None:
        super().__init__()
        if train:
            self.dataset = np.load('../20218078/X_train.npy', allow_pickle= True)
            with open('../20218078/y_train.txt', 'r') as f:
                self.labels = f.readlines()                
            #     self.label = [
            #         np.array(list(map(lambda x: int(x), line.rstrip('\n').split(',')[1:])))
            #             for line in f.readlines()
            #     ]
            # self.label = np.array(self.label)
        else:
            self.dataset = np.load('../20218078/X_test.npy', allow_pickle = True)
            with open('../20218078/y_test.txt', 'r') as f:
                self.labels = f.readlines()
            #     self.label = [
            #         np.array(list(map(lambda x: int(x), line.rstrip('\n').split(',')[1:])))
            #             for line in f.readlines()
            #     ]
            # self.label = np.array(self.label)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        txt = torch.LongTensor(self.dataset[index])
        # label = torch.LongTensor(self.label[index])
        label = self.labels[index]
        label = np.array(list(map(lambda x: int(x), label.rstrip('\n').split(',')[1:])))
        label = torch.LongTensor(label)

        return {'input': txt, 'label': label}

def collator(samples):
    labels = [sample['label'] for sample in samples]
    inputs = [sample['input'] for sample in samples]

    labels = torch.stack(labels)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first = True, padding_value = 1)
    padding_mask = (padded_inputs != 1).unsqueeze(2)

    return {
        'input': padded_inputs.contiguous(),
        'padding_mask': padding_mask.contiguous(),
        'label': labels.contiguous()
    }

batch_size = 32

train_dataset = MyDataset(train = True)
test_dataset = MyDataset(train = False)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collator)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collator)

def train(**kwargs):
    train_loader = kwargs['loader']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    criterion = kwargs['criterion']
    device = kwargs['device']
    epochs = kwargs['epochs']

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loss_mean = 0
        
        with tqdm(total = len(train_loader), desc = f"[Train] epoch {epoch + 1}") as pbar:
            for step, batch in enumerate(train_loader):
                input = batch['input']
                label = batch['label']
                padding_mask = batch['padding_mask']
                del batch

                input = input.to(device)
                label = label.to(device)
                padding_mask = padding_mask.to(device)

                optimizer.zero_grad()

                logits = model(input, padding_mask = padding_mask)
                loss = criterion(logits, label.float())

                loss.backward()
                optimizer.step()
                
                total_loss += loss

                loss_mean = total_loss / (step + 1)

                pbar.update(1)
                pbar.set_postfix_str(f"loss: {loss:.3f}, loss_mean: {loss_mean:.3f}")
                del logits

    print()
    print("[Test] Validation starts")
    auroc_macro, auroc_micro, auprc_macro, auprc_micro = validate(loader = test_loader, model = model, device = device)
    torch.save({
        "state_dict": model.state_dict(),
        "auroc_macro": auroc_macro,
        "auroc_micro": auroc_micro,
        "auprc_macro": auprc_macro,
        "auprc_micro": auprc_micro
    }, f'../20218078/model_epoch_{epoch+1}.pth')

def validate(**kwargs):
    valid_loader = kwargs['loader']
    model = kwargs['model']
    device = kwargs['device']

    model.eval()

    with tqdm(total = len(valid_loader), desc = f"[Test] ") as pbar:
        preds = []
        tgts = []
        for batch in valid_loader:
            input = batch['input']
            label = batch['label']
            padding_mask = batch['padding_mask']
            del batch
 
            input = input.to(device)
            label = label.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                logits = model(input, padding_mask = padding_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            preds.append(probs)
            tgts.append(label)

            pbar.update(1)

    preds = np.concatenate(preds, axis = 0)
    tgts = np.concatenate(tgts, axis = 0)

    survive = np.where(tgts == 0, True, False).all(axis = 0)
    survive = np.where(survive == False)[0]

    print("[Test] Validation results")

    auroc_micro = roc_auc_score(tgts, preds, average = 'micro')
    print(f"[Test] AUROC_micro: {auroc_micro:.4f}")
    auprc_micro = average_precision_score(tgts, preds, average = 'micro')
    print(f"[Test] AUPRC_micro: {auprc_micro:.4f}")

    preds = preds[:,survive]
    tgts = tgts[:,survive]

    auroc_macro = roc_auc_score(tgts, preds, average = 'macro')
    print(f"[Test] AUROC_macro: {auroc_macro:.4f}")
    auprc_macro = average_precision_score(tgts, preds, average = 'macro')
    print(f"[Test] AUPRC_macro: {auprc_macro:.4f}")
    print()

    return auroc_macro, auroc_micro, auprc_macro, auprc_micro


num_layers = 2
d_model = 128
d_ff = 512
h = 2

lr = 5e-4
epochs = 10

gpu = 0

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

model = make_model(72652, N = num_layers, d_model = d_model, d_ff = d_ff, h = h)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.BCEWithLogitsLoss()

model.to(device)

train(
    loader = train_loader,
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    device = device,
    epochs = epochs
)
