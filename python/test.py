import multiprocessing as mp
import pandas as pd
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('all-mpnet-base-v2')

def encode(arr):
  # print(f'encode: {arr}')
  ens = []
  for sen in arr:
    en = _model.encode(sen)
    print(f'encoded, prc: {mp.current_process()}, sen: {len(sen)}')
    ens.append(en)
    
  return ens
    

def load_data():
  X = pd.read_csv('/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/X.csv')
  y = pd.read_csv('/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/y.csv')
  
  return X, y
    

if __name__ == '__main__':
  # ctx = mp.get_context('spawn')
  # q = ctx.Queue()
  # p = ctx.Process(target=foo, args=(q,))
  # p.start()
  # print(q.get())
  # p.join()
  print('start loading data')
  X, y = load_data()
  print('complete loading data')
  
  print('start spawning context')
  ctx = mp.get_context('spawn')
  
  processes = []
  batch_size = 10000
  
  inputs = []
  for i in range(4):
    # inputs.append([SentenceTransformer('all-mpnet-base-v2'), X.iloc[i*batch_size:(i+1)*batch_size].values.tolist()])
    inputs.append(X.iloc[i*batch_size:(i+1)*batch_size].values.tolist())
  
  print('create pool')
  
  with ctx.Pool(4) as pool:
    res = pool.map(encode, X.iloc[:10].values.tolist())
    print(f'type: {type(res)}, {type(res[0])}, size: {len(res)}, {len(res[0][0])}')
    
    print(f'{res[0][0]}')
    
    N = len(res[0][0])
    columns = [str(i) for i in range(N)]
    
    df = pd.DataFrame(res)
    
    df = pd.DataFrame(df[0].values.tolist(), columns=columns)
    
    print(df.info())
    print(df.shape)
    print(df.head())
    
    # X_vectorize_miniLM_np = 
    
    
  print('complete')