# Model 4 - Fine tuning and training using ML model

import torch
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import numpy as np


print('Loading checkpoint data...')

# loading feature extracted data
data = torch.load(f='CIFAR10_features.pth')


X = data['features']
X = X.view(-1, 512) # to get the [50000, 512] dimension

y = data['labels']

print('\nConverting data| Tensor --> numpy...')
X_num = X.numpy()
y_num = y.numpy()


print('\nTrain-test split...')
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.30, random_state=42)


print('\nTraining Model...')
model = SVC(kernel="rbf", random_state=11)

model.fit(X_train,y_train)

print('\nPrediction Model...')
y_pred=model.predict(X_test)

print('\n')
report= classification_report(y_true=y_test,y_pred=y_pred)
print(report)

cnf = confusion_matrix(y_true=y_test,y_pred=y_pred)
print("\nConfusion matrix\n",cnf)
