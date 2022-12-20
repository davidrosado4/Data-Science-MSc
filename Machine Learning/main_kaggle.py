#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we perform predictions based on the fake_jobs dataset
"""

# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils import resample

from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

# Disable warnings
import warnings
warnings.filterwarnings('ignore')


# Import train dataset
df_train = pd.read_csv('nd_clear_train.csv', index_col = 0)

# Import test dataset
df_test = pd.read_csv('nd_clear_test.csv', index_col = 0)

# Import target
TARGET = pd.read_csv('fraudulent.csv', index_col = 0)


X = df_train.values.astype(np.float)
y = TARGET.values.astype(np.float)


print('Training algorithms...')


# concatenate our training data back together
auxX = pd.concat([df_train, TARGET], axis=1)

# separate minority and majority classes
not_fraud = auxX[auxX['fraudulent']==0]
fraud = auxX[auxX['fraudulent']==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

y_res = upsampled['fraudulent']
X_res = upsampled.drop('fraudulent', axis=1)


# train model
xgb_model = xgb.XGBClassifier(learning_rate=0.02, n_estimators=300)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 20, 40]
        }

clf = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=10, scoring='f1', cv=4)
clf.fit(X_res, y_res)


print("Average of the best f1-score in various folds during cross validation = ",clf.best_score_)
print("The best parameters found during k-fold cross validation is = ",clf.best_params_)