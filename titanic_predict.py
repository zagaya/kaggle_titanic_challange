from titanic import *

titanic = Titanic()

x_train, y_train, x_test = titanic.data_prep2()

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

knn_values = {
              'n_neighbors': [i for i in range(1, 20)],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
              }

knn = KNeighborsClassifier()
# start = time.time()
knn_acc = GridSearchCV(knn, param_grid=knn_values, scoring='accuracy').fit(x_train, y_train)
print('Best paremeters: {}'.format(knn_acc.best_params_))
print('Best score: {}'.format(knn_acc.best_score_))
# print('Running time: {} min'.format(round((time.time() - start)/60, 1)))

knn = knn_acc.best_estimator_.fit(x_train, y_train)

predictions = knn.predict(x_test).astype(int)

print('Accuracy knn: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))