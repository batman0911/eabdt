import os
import pandas as pd
import numpy as np 
import torch
from torcheval.metrics import BinaryAUROC

def load_label(folder, from_batch, to_batch):
  ys = []
  for i in range(from_batch, to_batch, 1):
    y = pd.read_csv(os.path.join(folder, f'y_{i}.csv'))
    ys.append(y)
  return pd.concat(ys, axis=0).to_numpy()


def load_vector(folder, from_batch, to_batch):
  Xs = []
  for i in range(from_batch, to_batch, 1):
    X = np.load(os.path.join(folder, f'X_{i}.npy'))
    Xs.append(X)
  return np.concatenate(Xs, axis=0)


def load_text(folder, from_batch, to_batch):
  Xs = []
  for i in range(from_batch, to_batch, 1):
    X = pd.read_csv(os.path.join(folder, f'X_{i}.csv'))
    Xs.append(X)
  return pd.concat(Xs, axis=0)


def metrics(model, criterion, X_test, y_test):
  correct_test = 0
  total_test = 0
  outputs_test = torch.squeeze(model(X_test))
  loss_test = criterion(outputs_test, y_test)

  total_test += y_test.size(0)
  correct_test += torch.eq(outputs_test.round(), y_test).sum()
  accuracy_test = 100 * correct_test/total_test
  
  metric = BinaryAUROC()
  metric.update(outputs_test, y_test)
  
  auc = metric.compute().item()
  
  return accuracy_test.item(), auc
  