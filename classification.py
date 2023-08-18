from util import load_vector, load_label, build_model
import os
import pandas as pd


data_folder = {
                'sbert_train': r'Data',
                'sbert_test': r'Data',
                'tfidf_train': r'Data',
                'tfidf_test': r'Data',
                }
# embeded_methods = ['sbert', 'tfidf']
# classify_methods = ['logistic_regression', 'svm', 'xgboost', 'neural_network']
embeded_methods = ['sbert']
classify_methods = ['logistic_regression', 'svm', 'xgboost']


train_from_batch = 0
train_to_batch = 3
test_from_batch = 3 
test_to_batch =  4


if 'Models' not in os.listdir():
    os.mkdir('Models')
if 'Figures' not in os.listdir():
    os.mkdir('Figures')


result = []
for embeding in embeded_methods:
    x_train = load_vector(data_folder[f'{embeding}_train'], train_from_batch, train_to_batch)
    y_train =  load_label(data_folder[f'{embeding}_train'], train_from_batch, train_to_batch)
    x_test = load_vector(data_folder[f'{embeding}_test'], test_from_batch, test_to_batch)
    y_test = load_label(data_folder[f'{embeding}_test'], test_from_batch, test_to_batch)
    
    for classifying in classify_methods:
        auc = build_model(embeding, classifying, x_train, y_train, x_test, y_test)
        result.append([embeding, classifying, auc])


result_df = pd.DataFrame(result)
result_df.columns = ['embeding', 'classifying', 'AUC']
result_df.to_excel('result.xlsx')




