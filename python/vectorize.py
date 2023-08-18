import pandas as pd   
import numpy as np
import sys
import os
import time
import multiprocessing
from multiprocessing import Process
from sentence_transformers import SentenceTransformer

dir_path = os.path.dirname(os.path.realpath(__file__))
# modelname = 'all-MiniLM-L6-v2'
modelname = 'all-mpnet-base-v2'


def load_batch(raw_path, batch_idx):
  filename = f'X_{batch_idx}.csv'
  return pd.read_csv(os.path.join(dir_path, raw_path, filename), dtype='string')


def vectorize(raw_path, clean_path, batch_list):
  process = multiprocessing.current_process()
  pid = process.pid
  model = SentenceTransformer(modelname)
  for batch_idx in batch_list:
    start_time = time.time()
    outname = f'X_{batch_idx}.npy'
    
    info = {
      'batch_index': batch_idx,
      'pid': pid
    }
  
    print(f'{info} \t| process ')
    
    print(f'{info} \t| start loading data')
    df = load_batch(raw_path, batch_idx)
    
    print(f'{info} \t| complete loading data')
    
    print(f'{info} \t| start vectorizing')
    df_vectorize = df.apply(lambda row : model.encode(row)[0], axis=1)
    np.save(os.path.join(dir_path, clean_path, outname), np.array(df_vectorize.values.tolist()))
    
    end_time = time.time()
    print(f'{info} \t| complete vectorize batch: {batch_idx} in {round(end_time - start_time, 2)} seconds')


if __name__ == '__main__':
  start_time = time.time()
  if len(sys.argv) < 1:
    raise "missing dataset"

  dataset = sys.argv[1]
  subset = sys.argv[2]
  total_batch = int(sys.argv[3])
  
  print(f'start processing {dataset}')
  
  raw_path = f'../data/raw/{dataset}/{subset}'
  clean_path = f'../data/vectorize/{dataset}/{subset}'

  num_process = 2
  num_batch_per_process = total_batch // num_process
  
  batch_process_indice = []
  
  for i in range(num_process):
    batch_from = i * num_batch_per_process
    batch_to = batch_from + num_batch_per_process
    tmp = [j for j in range(batch_from, batch_to, 1)]
    batch_process_indice.append(tmp)
    
  # add last batches to last process
  if total_batch % num_process != 0:
    tmp = [j for j in range(batch_process_indice[-1][-1] + 1, total_batch)]
    batch_process_indice[-1] = batch_process_indice[-1] + tmp
  
  # print(f'complete, {batch_process_indice}')
  
  p1 = Process(target=vectorize, args=(raw_path, clean_path, batch_process_indice[0]))
  p2 = Process(target=vectorize, args=(raw_path, clean_path, batch_process_indice[1]))
  
  p1.start()
  p2.start()
  
  p1.join()
  p2.join()
  
  end_time = time.time()
  print(f'complete vectorize for {total_batch} batch in {round(end_time - start_time, 2)} seconds')  
  