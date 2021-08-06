"""
Created on Fri Jul 30 18:44:01 2021

@author: maurya
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def rescaling(x):
    """Min-Max Normalisation or Rescaling

    Parameters
    ----------
    x : input matrix

    Returns
    -------
    Min-Max Normalised Matrix
    """
    min_x = x.min()
    max_x = x.max()
    range_x = max_x - min_x
    x_scaled = (x - min_x) / range_x
    return x_scaled


def mean_normal(x):
    """Mean Normalisation

    This method uses the mean of the observations in the transformation process

    Parameters
    ----------
    x : input matrix

    Returns
    -------
    A Standardized Matrix
    """
    mu = x.mean()
    min_x = x.min()
    max_x = x.max()
    range_x = max_x - min_x
    x_scaled = (x - mu) / range_x
    return x_scaled


def standardization(x):
    """Z-score Normalisation or Standardization

    This method uses Z-score or “standard score”

    z = (x - mu) / std

    mu : population mean

    std : population standard deviation

    Parameters
    ----------
    x : input matrix

    Returns
    -------
    A Standardized Matrix
    """
    mu = x.mean()
    std = x.std()
    x_scaled = (x - mu) / std
    return x_scaled


# getting the dataset
cancer = load_breast_cancer()
feature_set = cancer['feature_names']
x = cancer['data']
y = cancer['target'].reshape(cancer['target'].size, 1)  # 0 - malignant, 1 - benign
df = pd.DataFrame(np.c_[x, y], columns=np.append(cancer['feature_names'], ['target']))
x = cancer['data']
y = cancer['target']

"""
# visualising the data
sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df['target'])
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data = df)
sns.heatmap(df.corr(), annot=True)
"""

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
std_sc = StandardScaler()
x_train = std_sc.fit_transform(x_train)
x_test = std_sc.transform(x_test)

"""min_train = x_train.min()
range_train = (x_train - min_train).max()
x_train_scaled = (x_train - min_train) / range_train
min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_scaled = (x_test - min_test) / range_test"""
# x_train = rescaling(x_train)
# x_test = rescaling(x_test)

# training the model
classifier = SVC()
classifier.fit(x_train, y_train)

# evaluating the model
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

# optimizing parameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(x_train, y_train)
grid.best_params_
grid_predictions = grid.predict(x_test)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, grid_predictions))