from titanic2 import *
import titanic as t1
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

titanic = Titanic()
titanic.fillna()
titanic.data_prep()

train, labels, test = titanic.data_model()

# XGBoost Classifier
gbm = xgb.XGBClassifier(
    learning_rate=0.01,
    n_estimators=500,
    max_depth=3,
    min_child_weight=2,
    #      gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(train, labels)

predictions = gbm.predict(test).astype(int)

print('Accuracy xgboost: {:.4f}'.format(accuracy_score(predictions, t1.Titanic().solution()['Survived'].values)))
titanic.write_predictions(predictions,'xgb_180618_prep2_02.csv') # score 0.80382