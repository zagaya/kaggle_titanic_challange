import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from titanic import *

titanic = Titanic()

def sets():
    """
    Output:

        - train  : DataFrame
        - y_train: Series
        - test.  : DataFrame
    """

    def clean(df):
        """
        Input: DataFrame
        Output: DataFrame

        Insert new columns 'Child' and 'Group' into df and clean it.
        """

        cols_to_drop = [
            'Embarked',
            'Age',
            'Name',
            'Cabin',
            'Ticket',
            'Fare'
        ]

        extra_cols = df['Name'].str.extract(r'\w+,\s(?P<Child>\w+)', expand=False)
        df = pd.concat([df, extra_cols], axis=1)
        df['Child'] = np.where((df['Child'] == 'Master') |
                               ((df['Child'] == 'Miss') & (df['Age'] < 16)), 'child', 'adult')
        df[['Ticket', 'Fare']] = df[['Ticket', 'Fare']].astype(str)
        df['Group'] = df[['Ticket', 'Fare']].apply(lambda x: ''.join(x), axis=1)
        df['Pclass'] = df['Pclass'].map({1: 'first', 2: 'second', 3: 'third'})

        df.drop(cols_to_drop, axis=1, inplace=True)

        return df

    train = clean(pd.read_csv('train.csv'))
    test = clean(pd.read_csv('test.csv'))

    y_train = train['Survived']

    return train, y_train, test


def prepro_train(df):
    res = df.copy()

    res['friends_alive'] = 0
    res['friends_alive_male'] = 0
    for i in res.index:
        pas_id = res.loc[i, 'PassengerId']
        gr = res.loc[i, 'Group']
        sex = res.loc[i, 'Sex']
        friends_alive = res[(res['Group'] == gr) &
                            (res['PassengerId'] != pas_id) &
                            (res['Survived'] == 1)]
        res.loc[i, 'friends_alive'] = len(friends_alive)
        if sex == 'male':
            friends_alive_male = friends_alive[friends_alive['Sex'] == 'male']
            if len(friends_alive_male):
                res.loc[i, 'friends_alive_male'] = 1

    res['sons'] = 0
    res['sons_dead'] = 0
    for i in res.index:
        pas_id = res.loc[i, 'PassengerId']
        gr = res.loc[i, 'Group']
        adult_woman = (res.loc[i, 'Child'] == 'adult') and (res.loc[i, 'Sex'] == 'female')
        if adult_woman:
            sons = res[(res['Group'] == gr) &
                       (res['Child'] == 'child')]
            sons_dead = sons[sons['Survived'] == 0]
            res.loc[i, 'sons'] = len(sons)
            res.loc[i, 'sons_dead'] = len(sons_dead)

    return res


def prepro_real(df1, df2):
    res = df1.copy()

    res['friends_alive'] = 0
    res['friends_alive_male'] = 0
    for i in res.index:
        pas_id = res.loc[i, 'PassengerId']
        gr = res.loc[i, 'Group']
        sex = res.loc[i, 'Sex']
        friends_alive = df2[(df2['Group'] == gr) &
                            (df2['PassengerId'] != pas_id) &
                            (df2['Survived'] == 1)]
        res.loc[i, 'friends_alive'] = len(friends_alive)
        if sex == 'male':
            friends_alive_male = friends_alive[friends_alive['Sex'] == 'male']
            if len(friends_alive_male):
                res.loc[i, 'friends_alive_male'] = 1

    res['sons'] = 0
    res['sons_dead'] = 0
    for i in res.index:
        pas_id = res.loc[i, 'PassengerId']
        gr = res.loc[i, 'Group']
        adult_woman = (res.loc[i, 'Child'] == 'adult') and (res.loc[i, 'Sex'] == 'female')
        if adult_woman:
            sons = df2[(df2['Group'] == gr) &
                       (df2['Child'] == 'child')]
            sons_dead = sons[sons['Survived'] == 0]
            res.loc[i, 'sons'] = len(sons)
            res.loc[i, 'sons_dead'] = len(sons_dead)

    return res

X_train, y_train, X_real = sets()

X_train = prepro_train(X_train)
X_real = prepro_real(X_real, X_train)

info_tr = X_train[['PassengerId', 'SibSp', 'Parch', 'Group',
                   'Survived', 'Sex', 'Child', 'Pclass']]
X_train.drop(['PassengerId',
              'Group',
              'Survived'], axis=1, inplace=True)
X_train = pd.get_dummies(X_train)


info_re = X_real[['PassengerId', 'SibSp', 'Parch',
                  'Group', 'Sex', 'Child', 'Pclass']]
X_real.drop(['PassengerId',
             'Group'], axis=1, inplace=True)
X_real = pd.get_dummies(X_real)


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
    scale_pos_weight=1).fit(X_train, y_train)

predictions = gbm.predict(X_real).astype(int)

print('Accuracy xgboost: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))

knn_values = {
              'n_neighbors': [i for i in range(1, 20)],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
              }

knn = KNeighborsClassifier()
start = time.time()
knn_acc = GridSearchCV(knn, param_grid=knn_values, scoring='accuracy').fit(X_train, y_train)
print('Best paremeters: {}'.format(knn_acc.best_params_))
print('Best score: {}'.format(knn_acc.best_score_))
print('Running time: {} min'.format(round((time.time() - start)/60, 1)))

knn = knn_acc.best_estimator_.fit(X_train, y_train)

predictions = knn.predict(X_real).astype(int)

print('Accuracy knn: {:.4f}'.format(accuracy_score(predictions, titanic.solution()['Survived'].values)))