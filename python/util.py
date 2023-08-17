import os
import pandas as pd
import numpy as np 


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
  