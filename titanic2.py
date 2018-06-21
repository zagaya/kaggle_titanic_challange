import pandas as pd
import numpy as np

class Titanic():

    def __init__(self):
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

        self.passenger_id_train = train['PassengerId']
        self.passenger_id_test = test['PassengerId']

        total = pd.concat([train, test], axis=0)

        title = total.Name.str.findall(r'^[\w\W]*, (\w*)').map(lambda s: s[0])

        title_replace = {'Master': 'Children',
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

        title.replace(title_replace, inplace=True)

        total['Title'] = title
        total.loc[((total['Age'] < 16) & (total['Title'] == 'Miss')), 'Title'] = 'Children'
        total['Surname'] = total.Name.str.findall(r'^(.*),')
        total['Surname'] = total['Surname'].apply(lambda x: x[0])

        total.reset_index(drop=True, inplace=True)
        self._train = train
        self._test = test
        self._total = total

    @property
    def total(self):
        return self._total.copy()

    @total.setter
    def total(self, new_total):
        self._total = new_total

    @property
    def train(self):
        return self._total.loc[self.passenger_id_train-1,:].drop(['Survived'],axis=1)

    @property
    def labels(self):
        return self._total.loc[self.passenger_id_train-1,'Survived']

    @property
    def test(self):
        return self._total.loc[self.passenger_id_test-1,:].drop(['Survived'],axis=1)

    def write_predictions(self, predictions, filename='titanic_pred.csv'):
        pred = pd.DataFrame({'PassengerId': self.passenger_id_test.values, 'Survived': predictions})
        pred.set_index('PassengerId', inplace=True)
        pred.to_csv(filename)

    def fillna(self):

        total = self.total

        # Filling Embarked nans with majority case 'S'
        total.loc[total['Embarked'].isnull(), 'Embarked'] = 'S'

        # Filling nan for Fare and Age based on Pclass and Title medians respectively
        # Age based on Title
        for title in total['Title'].unique():
            total.loc[((total['Title'] == title) &
                       (total['Age'].isnull())), 'Age'] = round(total[total['Title'] == title]['Age'].median())

        # Fare based on Pclass
        for pclass in total['Pclass'].unique():
            total.loc[((total['Pclass'] == pclass) &
                       (total['Fare'].isnull())), 'Fare'] = round(total[total['Pclass'] == pclass]['Fare'].median())

        self.total = total

    def data_prep(self):

        total = self.total

        total['Ticket'] = total['Ticket'].apply(lambda x: x[:-1] + 'X')
        total['Group'] = total.apply(lambda x: '-'.join([x['Surname'], x['Ticket'], str(x['Fare'])]), axis=1)
        total['Children'] = total['Age'].apply(lambda x: int(x < 16))
        total['Women'] = total['Sex'].apply(lambda x: int(x == 'female'))
        total['WCG']=0
        total['People'] = 1
        temp = total[['Group', 'Women', 'Children', 'People', 'Survived']].groupby(['Group']).sum()
        groups = temp[temp['People'] > 1].index.unique()
        features = ['WomenSurv','WomenDead','ChildSurv','ChildDead']

        for f in features:
            total[f]=0

        for group in groups:
            total.loc[total['Group']==group,'WCG']=1
            temp = total[total['Group'] == group]
            total.loc[total['Group'] == group, 'WomenSurv'] = temp.apply(
                lambda x: int((x['Sex'] == 'female') & (x['Survived'] == 1)), axis=1)
            total.loc[total['Group'] == group, 'WomenDead'] = temp.apply(
                lambda x: int((x['Sex'] == 'female') & (x['Survived'] == 0)), axis=1)
            total.loc[total['Group'] == group, 'ChildSurv'] = temp.apply(
                lambda x: int((x['Age'] < 16) & (x['Survived'] == 1)), axis=1)
            total.loc[total['Group'] == group, 'ChildDead'] = temp.apply(
                lambda x: int((x['Age'] < 16) & (x['Survived'] == 0)), axis=1)

        # Cabin has too many nans, drop it
        total.drop(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch',
                    'SibSp', 'Ticket', 'Title', 'Surname', 'Group', 'People', 'Sex'], axis=1, inplace=True)

        # One-hot labeling
        # # Label sex
        # total.loc[total['Sex'] == 'male', 'Sex'] = 1
        # total.loc[total['Sex'] == 'female', 'Sex'] = 0

        self.total = total

    def data_model(self):
        x_train, y_train, x_test = self.train, self.labels, self.test
        return x_train, y_train, x_test

titanic = Titanic()
#
titanic.fillna()
# titanic.data_prep()
#
# total = titanic.total
#
#
#
# print('yo')

