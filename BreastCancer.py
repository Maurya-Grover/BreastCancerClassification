"""
Created on Fri Jul 30 18:44:01 2021

@author: maurya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# getting the dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
feature_set = cancer['feature_names']
x = cancer['data']
y = cancer['target'].reshape(cancer['target'].size, 1) # 0 - malignant, 1 - benign
df = pd.DataFrame(np.c_[x, y], columns=np.append(cancer['feature_names'], ['target']))
x = pd.DataFrame(x, columns=feature_set)
y = pd.DataFrame(y, columns=['target'])

# visualising the data
sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df['target'])
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data = df)
sns.heatmap(df.corr(),annot=True)

#splitting the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
"""std_sc = StandardScaler()
x_train = std_sc.fit_transform(x_train)
x_test = std_sc.transform(x_test)"""

#training the model
from sklearn.svm import SVC
classifier = SVC(random_state=0)