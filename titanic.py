import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class Titanic():

    def __init__(self):
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

        train['train'] = 1
        test['train'] = 0

        self._passenger_id = test['PassengerId']

        total = pd.concat([train, test], axis=0)

        title = total.Name.str.findall(r'^[\w\W]*, (\w*)').map(lambda s: s[0])

        social_class = {'Master': 'Children',
                        'Don': 'Officer',
                        'Rev': 'Officer',
                        'Dr': 'Officer',
                        'Mme': 'Miss',
                        'Ms': 'Miss',
                        'Major': 'Officer',
                        'Lady': 'Mrs',
                        'Sir': 'Mrs',
                        'Mlle': 'Miss',
                        'Col': 'Officer',
                        'Capt': 'Officer',
                        'the': 'Mrs',
                        'Jonkheer': 'Officer',
                        'Dona': 'Mrs'}

        title.replace(social_class, inplace=True)

        total['Title'] = title
        total['Surname'] = total.Name.str.findall(r'^(.*),')
        total['Surname'] = total['Surname'].apply(lambda x: x[0])

        # total.drop('Name', axis=1, inplace=True)
        total.reset_index(drop=True, inplace=True)
        self._train = train
        self._test = test
        self._total = total

    @property
    def train(self):
        return self._train.copy()

    @property
    def test(self):
        return self._test.copy()

    @property
    def total(self):
        return self._total.copy()

    def write_predictions(self, predictions, filename='titanic_pred.csv'):
        pred = pd.DataFrame({'PassengerId': self._passenger_id.values, 'Survived': predictions})
        pred.set_index('PassengerId', inplace=True)
        pred.to_csv(filename)

    def solution(self):
        # Get the solution labels to compute the accuracy
        test = self.test.copy()

        official_record = pd.read_excel('titanic3.xls')
        official_record.drop(index=[725, 925], inplace=True)
        official_record = official_record[['survived', 'name', 'age']]

        official_record['name'] = official_record['name'].str.replace('"', '')
        test['Name'] = test['Name'].str.replace('"', '')

        official_record = official_record.rename(columns={'name': 'Name', 'age': 'Age', 'survived': 'Survived'})
        solution = test[['Name', 'PassengerId']].merge(official_record, on=['Name'], how='inner')

        solution = solution[['PassengerId', 'Survived']]

        self._solution = solution

        return self._solution

    # data prep2
    def data_prep2(self):

        total = self.total

        # Family engineering
        # total.loc[total['SibSp']>2,'SibSp']=3
        # total.loc[total['Parch']>2,'Parch']=3

        total['Family'] = total['SibSp'] + total['Parch'] + 1
        total['isAlone'] = 0
        total.loc[total['Family'] == 1, 'isAlone'] = 1
        total.drop(['SibSp', 'Parch'], axis=1, inplace=True)

        total.loc[((total['Age'] < 14) & (total['Title'] == 'Miss')), 'Title'] = 'GirlChildren'

        # Cabin has too many nans, drop it
        total.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

        # filling Embarked nans with majority case 'S'
        total.loc[total['Embarked'].isnull(), 'Embarked'] = 'S'

        # fill nan for fare and age based on Pclass and Title medians respectively

        # Age based on Title
        for title in total['Title'].unique():
            total.loc[((total['Title'] == title) &
                       (total['Age'].isnull())), 'Age'] = round(total[total['Title'] == title]['Age'].median())

        # Fare based on Pclass
        for pclass in total['Pclass'].unique():
            total.loc[((total['Pclass'] == pclass) &
                       (total['Fare'].isnull())), 'Fare'] = round(total[total['Pclass'] == pclass]['Fare'].median())

        def survival_rate_feature(total, feature, new_feature, intervals):

            total[new_feature] = 0

            for i, j in intervals:
                temp = total[(total[feature] >= i) & (total[feature] < j) & (total['train'] == 1)]['Survived']
                n_passengers = temp.shape[0]
                rate = temp.sum() / n_passengers
                total.loc[((total[feature] >= i) & (total[feature] < j)), new_feature] = round(rate, 2)
            # total[new_feature] = total[new_feature] / total[new_feature].max()
            return total.drop(feature, axis=1)

        # 3 fasce di prezzo 1 - fino a 10, 2 - da 10 a 40, 3 piu di 40
        # total.loc[total['Fare']<=10,'Fare']=1
        # total.loc[((total['Fare']>10) & (total['Fare']<=40)),'Fare']=2
        # total.loc[total['Fare']>40,'Fare']=3

        fare_intervals = [(0, 10), (10, 25), (25, 50), (50, 1000)]
        total = survival_rate_feature(total, 'Fare', 'FareRate', fare_intervals)

        # 5 fasce di eta'
        age_intervals = [(0, 1), (1, 10), (10, 25), (25, 50), (60, 100)]
        total = survival_rate_feature(total, 'Age', 'AgeRate', age_intervals)

        #     total.loc[:,'Age'] = total.loc[:,'Age'].astype(int)
        #     total.loc[:,'Fare'] = total.loc[:,'Fare'].astype(int)

        # Label sex
        total.loc[total['Sex'] == 'male', 'Sex'] = 1
        total.loc[total['Sex'] == 'female', 'Sex'] = 0

        total.drop(['Surname', 'Name'], axis=1, inplace=True)

        # Labels Embark and Title
        features = ['Embarked', 'Title']

        for f in features:
            dummy = pd.get_dummies(total[f], prefix=f)
            total.drop(f, axis=1, inplace=True)

            total = pd.concat((total, dummy), axis=1)



        train_set = total[total['train'] == 1]
        test_set = total[total['train'] == 0]

        y_train = train_set['Survived']
        x_train = train_set.drop(['Survived', 'train'], axis=1)
        x_test = test_set.drop(['Survived', 'train'], axis=1)

        return x_train, y_train, x_test

    def data_prep3(self):

        total = self.total

        # Family feature engineering
        total['Family'] = total['SibSp'] + total['Parch'] + 1
        total['isAlone'] = 0
        total.loc[total['Family'] == 1, 'isAlone'] = 1
        total.drop(['SibSp'], axis=1, inplace=True)

        total.loc[((total['Age'] < 14) & (total['Title'] == 'Miss')), 'Title'] = 'GirlChildren'

        # filling Embarked nans with majority case 'S'
        total.loc[total['Embarked'].isnull(), 'Embarked'] = 'S'
        #
        # Fill nan for Fare and Age based on Pclass and Title medians respectively

        # Age based on Title
        for title in total['Title'].unique():
            total.loc[((total['Title'] == title) &
                       (total['Age'].isnull())), 'Age'] = round(total[total['Title'] == title]['Age'].median())

        # Fare based on Pclass
        for pclass in total['Pclass'].unique():
            total.loc[((total['Pclass'] == pclass) &
                       (total['Fare'].isnull())), 'Fare'] = round(total[total['Pclass'] == pclass]['Fare'].median())

        def survival_rate_feature(total, feature, new_feature, intervals):
            total[new_feature] = 0

            for i, j in intervals:
                temp = total[(total[feature] >= i) & (total[feature] < j) & (total['train'] == 1)]['Survived']
                n_passengers = temp.shape[0]
                rate = temp.sum() / n_passengers
                total.loc[((total[feature] >= i) & (total[feature] < j)), new_feature] = round(rate, 2)
            # total[new_feature] = total[new_feature] / total[new_feature].max()

            return total.drop(feature, axis=1)

        # total['Group'] = 0
        # group_label = 0
        # for surname in total['Surname'].unique():
        #     same_name = total[((total['Surname'] == surname) & (total['Group'] == 0) & (total['isAlone'] == 0))]
        #     group_label = group_label + 1
        #
        #     for ticket in same_name['Ticket'].unique():
        #         total.loc[(((total['Surname'] == surname) | (total['Ticket'] == ticket))
        #                    & (total['isAlone'] == 0)), 'Group'] = group_label
        #
        # for group_label in total['Group'].unique():
        #     temp = total[(total['Group']==group_label) & (total['train']==1)]['Survived']
        #     n_passengers = temp.shape[0]
        #     print('label: {} \t npas: {} \t survived: {}')
        #
        #     if n_passengers:
        #         rate = temp.sum()/n_passengers
        #         total.loc[total['Group']==group_label,'GroupRate'] = round(rate,2)
        #     else:
        #         rate = 0.5
        #         total.loc[total['Group'] == group_label, 'GroupRate'] = round(rate, 2)
        total['Group'] = 0
        group_label = 0
        for ticket in total['Ticket'].unique():

            n_ticket = (total['Ticket'] == ticket).sum()

            if n_ticket > 1:
                group_label = group_label + 1
                total.loc[(total['Ticket'] == ticket), 'Group'] = group_label

        # for group_label in total['Group'].unique():
        #     temp = total[(total['Group']==group_label) & (total['train']==1)]['Survived']
        #     n_passengers = temp.shape[0]
        #     print('label: {} \t npas: {} \t survived: {}')
        #
        #     if n_passengers:
        #         rate = temp.sum()/n_passengers
        #         total.loc[total['Group']==group_label,'GroupRate'] = round(rate,2)
        #     else:
        #         rate = 0.5
        #         total.loc[total['Group'] == group_label, 'GroupRate'] = round(rate, 2)

        # total.drop('Group', axis=1)
        # total['Group'] = total['Group'].apply(lambda x: str(x))
        # for group_label in total['Group'].unique():

        total['Pclass'] = total['Pclass'].apply(lambda x: str(x))

        fare_intervals = [(0, 10), (10, 25), (25, 50), (50, 1000)]
        total = survival_rate_feature(total, 'Fare', 'FareRate', fare_intervals)

        # 5 fasce di eta'
        age_intervals = [(0, 1), (1, 10), (10, 25), (25, 50), (60, 100)]
        total = survival_rate_feature(total, 'Age', 'AgeRate', age_intervals)

        # Label sex
        total.loc[total['Sex'] == 'male', 'Sex'] = 1
        total.loc[total['Sex'] == 'female', 'Sex'] = 0

        # Cabin has too many nans, drop it
        total.drop(['Cabin', 'Ticket', 'PassengerId', 'Surname', 'Name'], axis=1, inplace=True)

        #
        # Labels
        total = pd.get_dummies(total)

        train_set = total[total['train'] == 1]
        test_set = total[total['train'] == 0]

        y_train = train_set['Survived']
        x_train = train_set.drop(['Survived', 'train'], axis=1)
        x_test = test_set.drop(['Survived', 'train'], axis=1)

        return x_train, y_train, x_test


# titanic = Titanic()
#
# x_train, y_train, x_test = titanic.data_prep3()
