from titanic import *

titanic = Titanic()

x_train, y_train, x_test = titanic.data_prep3()

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
    scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test).astype(int)

print('Accuracy xgboost: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))
# titanic.write_predictions(predictions,'xgb_180618_prep2_02.csv') # score 0.80382
