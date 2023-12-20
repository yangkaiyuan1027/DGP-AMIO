import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter, Linear
from sklearn import metrics
from models import triGAT,triGAT_without_edge_feature, GAT,GAT_without_edge_feature , GCN
import scanpy as sc


#Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Loss():
    def __init__(self, y, idx):
        self.y = y
        idx = np.array(idx)

        self.y_pos = y[y == 1]
        self.y_neg = y[y == 0]

        self.pos = idx[y.cpu() == 1]
        self.neg = idx[y.cpu() == 0]

    def __call__(self, out):
        loss_p = F.binary_cross_entropy_with_logits(
            out[self.pos].squeeze(), self.y_pos)
        loss_n = F.binary_cross_entropy_with_logits(
            out[self.neg].squeeze(), self.y_neg)
        loss = loss_p + loss_n
        return loss

def evalAUC(model, X, A1, A2, A3, edge_feature, y, mask, logits=None):
    assert(model is not None or logits is not None)
    if model is not None:
        model.eval()
        with torch.no_grad():
            logits = model(X, A1, A2, A3, edge_feature)
            logits = logits[mask]
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()
    y = y.cpu().numpy()
    auc = metrics.roc_auc_score(y, probs)
    return auc


def evaluate(model, X, A1, A2, A3, edge_feature, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, A1, A2, A3, edge_feature)
        logits = logits[mask]
        preds = (torch.sigmoid(logits) > 0.5).to(torch.float32)[:, 0]
        correct = torch.sum(preds == y)
        return correct.item() * 1.0 / len(y)

def model_train(params, X, A1, A2, A3, edge_feature, train_y, train_idx,val_y, val_idx,savepath='', DEVICE='cpu'):
    epochs = 2000
    patience = 100
    model = triGAT(in_feats=X.shape[1], edge_feats=edge_feature.shape[1],**params)

    earlystopping=EarlyStopping(patience,verbose=True,path=savepath)
    model.to(DEVICE)
    X = X.to(DEVICE)
    A1 = A1.to(DEVICE)
    A2 = A2.to(DEVICE)
    A3 = A3.to(DEVICE)
    edge_feature=edge_feature.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_y = val_y.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fnc = Loss(train_y, train_idx)
    val_loss_fnc = Loss(val_y, val_idx)

    iterable = tqdm(range(epochs))
    for i in iterable:
        model.train()
        logits = model(X, A1, A2, A3, edge_feature)

        optimizer.zero_grad()
        loss = loss_fnc(logits)
        loss.backward()
#         train_losses.append(float(loss))
        optimizer.step()

        logits = logits.detach()
        val_loss = val_loss_fnc(logits)
#         val_losses.append(float(val_loss))
        train_auc = evalAUC(None, 0, 0, 0, 0, 0, train_y, 0, logits[train_idx])
        val_auc = evalAUC(None, 0, 0, 0, 0, 0, val_y, 0, logits[val_idx])
        earlystopping(val_loss,model)

        #tqdm.set_description(iterable, desc='Loss: %.4f ; Val Loss %.4f ; Train AUC %.4f. Validation AUC: %.4f' % (
        #    loss, val_loss, train_auc, val_auc))
        if earlystopping.early_stop:
            print('Early stopping')
            break

    score = evalAUC(model, X, A1, A2, A3, edge_feature, val_y, val_idx)
    #print(f'Last validation AUC: {val_auc}')

    return model

def test(model, X, A1, A2, A3, edge_feature, test_ds=None,DEVICE='cpu'):
    model.to(DEVICE).eval()
    X = X.to(DEVICE)
    A1 = A1.to(DEVICE)
    A2 = A2.to(DEVICE)
    A3 = A3.to(DEVICE)
    edge_feature=edge_feature.to(DEVICE)

    with torch.no_grad():
        logits = model(X, A1, A2, A3, edge_feature)
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()

    if test_ds is not None:
        test_idx, test_y = test_ds
        test_y = test_y.cpu().numpy()
        auc = metrics.roc_auc_score(test_y, probs[test_idx])
        return probs, auc
    return probs
