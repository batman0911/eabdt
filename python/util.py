import os
import pandas as pd
import numpy as np 
import torch
from torcheval.metrics import BinaryAUROC
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import torch.nn as nn
import torch.optim as optim

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


class NeuralNetwork_SBert(nn.Module):
    def __init__(self):
        super(NeuralNetwork_SBert, self).__init__()
        self.layer1 = nn.Linear(384, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)  # 2 classes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


def roc_curve_plot(model_name, auc, fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name.upper()}')
    plt.legend(loc='lower right')
    plt.savefig(fr'Figures/{model_name}_roc_curve.png')
    
    
def build_model(embeding, classifying, x_train, y_train, x_test, y_test):
    if classifying == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=1000, tree_method='gpu_hist')
        model.fit(x_train, y_train)
        with open(fr'Models/{classifying}_{embeding}.p', 'wb') as model_file:
            pickle.dump(model, model_file)
        y_pred = model.predict_proba(x_test)[:, 1]
        # with open(fr'Models/xgboost_{embeding}p', 'rb') as model_file:
        #     loaded_model = pickle.load(model_file)
        
    if classifying == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)        
        joblib.dump(model, fr'Models/{classifying}_{embeding}.p')
        y_pred = model.predict_proba(x_test)[:, 1]
        
    if classifying == 'svm':
        model = SVC(kernel='linear', probability=True)
        model.fit(x_train, y_train)
        joblib.dump(model, fr'Models/{classifying}_{embeding}.p')
        y_pred = model.predict_proba(x_test)[:, 1]

    if classifying == 'neural_network':
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        if embeding == 'sbert':
            model = NeuralNetwork_SBert()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            epochs = 1000
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            # Calculate predicted probabilities for class label '1'
            with torch.no_grad():
                model.eval()
                y_pred = torch.softmax(outputs, dim=1)[:, 1]
    
    acc = accuracy_score(y_test, np.round_(y_pred))  
    auc = roc_auc_score(y_test, y_pred)
    print(f"{classifying}_{embeding} Accuracy: {acc}, AUC: {auc}")    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_curve_plot(f'{classifying}_{embeding}', auc, fpr, tpr)
    return acc, auc
