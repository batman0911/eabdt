import pandas as pd   
import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer

dir_path = os.path.dirname(os.path.realpath(__file__))
raw_path = '../data/raw/Movies_and_TV'
clean_path = '../data/vectorize/Movies_and_TV'

modelname = 'all-MiniLM-L6-v2'

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise "missing batch index"
  
  batch_idx = sys.argv[1]
  filename = f'X_{batch_idx}.csv'
  outname = f'X_{batch_idx}.npy'
  
  print(f'process batch: {batch_idx}')
  
  print(f'start loading data')
  df = pd.read_csv(os.path.join(dir_path, raw_path, filename), header=None, dtype='string')
  
  print(f'def type: {type(df)}')
  print(df.info())
  print(df.head())
  
  print(f'complete loading data')
  
  print(f'start loading model: {modelname}')
  model = SentenceTransformer(modelname)
  
  print(f'start vectorizing')
  df_vectorize = df[0].apply(lambda row : model.encode(row))
  np.save(os.path.join(dir_path, clean_path, outname), np.array(df_vectorize.tolist()))
  
  print(f'complete vectorizing')