from util import load_vector, load_label, build_model
import os
import pandas as pd


data_folder = {
                'sbert_train': '/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/384/mix/training_set',
                'sbert_test': '/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/384/mix/testing_set',
                'tfidf_train': '../data',
                'tfidf_test': '../data',
                }
# embeded_methods = ['sbert', 'tfidf']
# classify_methods = ['logistic_regression', 'svm', 'xgboost', 'neural_network']
embeded_methods = ['sbert']
# classify_methods = ['logistic_regression', 'svm', 'xgboost']
classify_methods = ['xgboost']


train_from_batch = 0
train_to_batch = 3000
test_from_batch = 0 
test_to_batch =  400


if 'Models' not in os.listdir():
    os.mkdir('Models')
if 'Figures' not in os.listdir():
    os.mkdir('Figures')


result = []
for embeding in embeded_methods:
    x_train = load_vector('/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/384/mix/training_set', train_from_batch, train_to_batch)
    y_train =  load_label('/home/linhnm/msc_code/big_data_mining/eabdt/data/raw/mix/training_set', train_from_batch, train_to_batch)
    x_test = load_vector('/home/linhnm/msc_code/big_data_mining/eabdt/data/vectorize/384/mix/testing_set', test_from_batch, test_to_batch)
    y_test = load_label('/home/linhnm/msc_code/big_data_mining/eabdt/data/raw/mix/testing_set', test_from_batch, test_to_batch)
    
    for classifying in classify_methods:
        acc, auc = build_model(embeding, classifying, x_train, y_train, x_test, y_test)
        result.append([embeding, classifying, acc, auc])


result_df = pd.DataFrame(result)
result_df.columns = ['embeding', 'classifying', 'ACC', 'AUC']
result_df.to_excel('result.xlsx')




