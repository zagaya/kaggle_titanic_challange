from titanic import *

import warnings
warnings.filterwarnings("ignore")

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

# print('Accuracy xgboost: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))
# titanic.write_predictions(predictions,'xgb_180618_prep2_02.csv') # score 0.80382

# knn_values = {
#               'n_neighbors': [i for i in range(1, 20)],
#               'weights': ['uniform', 'distance'],
#               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#               }

# knn = KNeighborsClassifier()
# # start = time.time()
# knn_acc = GridSearchCV(knn, param_grid=knn_values, scoring='accuracy').fit(x_train, y_train)
# print('Best paremeters: {}'.format(knn_acc.best_params_))
# print('Best score: {}'.format(knn_acc.best_score_))
# # print('Running time: {} min'.format(round((time.time() - start)/60, 1)))
#
# knn = knn_acc.best_estimator_.fit(x_train, y_train)
#
# predictions = knn.predict(x_test).astype(int)
#
# print('Accuracy knn: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))

# xgb_values = dict(learning_rate=[0.01],
#                   n_estimators=[300, 500, 700],
#                   max_depth=[2, 3, 4, 5],
#                   min_child_weight=[1, 2, 3, 4],
#                   gamma=[0.1, 0.9, 0.5],
#                   subsample=[0.8],
#                   colsample_bytree=[0.8],
#                   objective=['binary:logistic'],
#                   nthread=[-1],
#                   scale_pos_weight=[0.5, 1])
#
# gmb2 = xgb.XGBClassifier()
# # # start = time.time()
# gmb2_grid = GridSearchCV(gmb2, param_grid=xgb_values, scoring='accuracy', verbose=5).fit(x_train, y_train)
# print('Best paremeters: {}'.format(gmb2_grid.best_params_))
# print('Best score: {}'.format(gmb2_grid.best_score_))
# # # print('Running time: {} min'.format(round((time.time() - start)/60, 1)))
#
# predictions2 = gmb2_grid.best_estimator_.fit(x_train, y_train).predict(x_test).astype(int)

gmb2 = xgb.XGBClassifier(learning_rate=0.01,
                  n_estimators=300,
                  max_depth=2,
                  min_child_weight=1,
                  gamma=1,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  nthread=-1,
                  objective='binary:logistic',
                  scale_pos_weight=0.6)

predictions2 = gmb2.fit(x_train, y_train).predict(x_test).astype(int)

print('Accuracy xgboost prep2: {:.4f}\nAccuracy xgboost gridsc: {:.4f}'.
      format(accuracy_score(predictions, titanic.solution()['Survived'].values),
             accuracy_score(predictions2, titanic.solution()['Survived'].values)))

# titanic.write_predictions(predictions2,'xgb_200618_prep2new_02.csv') # score 0.78