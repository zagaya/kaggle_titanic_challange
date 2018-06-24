import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

# TODO: predire eta' femmine basate su Parch per trovare le bambine
# TODO: studiare  le relazioni tra passeggeri, categorie biglietti e biglietti vicini. > Correggerere errori famSize
# TODO: didivi i passeggeri in gruppi e poi in base a quanti adulti e il valore di Parch predirre l'eta' delle miss

def load_data():
    """
    This function loads and prepares the 'total' attribute, which is used for processing the features and then is
    split to feed the predictors.
    """

    # CSVs loading
    train_csv = pd.read_csv('train.csv')
    test_csv = pd.read_csv('test.csv')

    # Data sets are concatenated to perform data processing operations at once
    total = pd.concat([train_csv, test_csv], axis=0)

    # PassengerId is set as index
    total = total.set_index('PassengerId')

    return total


def solution():
    '''
    The official list of Titanic passengers is available online and we can extract the solution of our problem from it.
    This can be used to compute a final scoring without using Kaggle evaluator. Kaggle evaluates only
    a subset of the total test set, therefore the scoring obtained with this solution will differ slightly.
    '''

    # Get the solution labels to compute the accuracy
    test = pd.read_csv('test.csv')

    official_record = pd.read_excel('titanic3.xls')
    official_record.drop(index=[725, 925], inplace=True)
    official_record = official_record[['survived', 'name', 'age']]

    official_record['name'] = official_record['name'].str.replace('"', '')
    test['Name'] = test['Name'].str.replace('"', '')

    official_record = official_record.rename(columns={'name': 'Name', 'age': 'Age', 'survived': 'Survived'})
    solution_series = test[['Name', 'PassengerId']].merge(official_record, on=['Name'], how='inner')

    solution_series = solution_series[['PassengerId', 'Survived']]

    return solution_series


class Titanic:

    def __init__(self, data=load_data()):

        self.passenger_id_train = data[~data['Survived'].isnull()].index
        self.passenger_id_test = data[data['Survived'].isnull()].index
        self.predictions = pd.Series(data=0, index=data[data['Survived'].isnull()].index)
        self._total = data  # All passengers data set - train and test concatenated for processing purposes

    @property
    def total(self):
        return self._total.copy()

    @total.setter
    def total(self, new_total):
        self._total = new_total

    @property
    def train(self):
        return self._total.loc[self.passenger_id_train,:].drop(['Survived'],axis=1)

    @property
    def labels(self):
        return self._total.loc[self.passenger_id_train,'Survived']

    @property
    def test(self):
        return self._total.loc[self.passenger_id_test,:].drop(['Survived'],axis=1)

    def name_features(self):
        """
        This method corrects some sparse passengers titles, used for age estimation later, and adds the 'Title' and
        'Surname' columns to the 'total' data frame
        """
        data = self.total

        # Replacing titles
        title = data.Name.str.findall(r'^[\w\W]*, (\w*)').map(lambda s: s[0]) # TODO: setting . title selection
        title_replace = dict(Master='Children', Don='Officer', Rev='Officer', Dr='Officer', Mme='Miss', Ms='Miss',
                             Major='Officer', Lady='Mrs', Sir='Mrs', Mlle='Miss', Col='Officer', Capt='Officer',
                             the='Mrs', Jonkheer='Officer', Dona='Mrs')
        title.replace(title_replace, inplace=True)
        data['Title'] = title

        # Girls less than 16 have title Miss, but is important to consider them Children
        data.loc[((data['Age'] < 16) & (data['Title'] == 'Miss')), 'Title'] = 'Children'

        # Surnames are extracted to perform analysis on groups of related people
        data['Surname'] = data.Name.str.findall(r'^(.*),')
        data['Surname'] = data['Surname'].apply(lambda x: x[0])

        return Titanic(data)

    def fillna(self):

        data = self.total

        # Filling Embarked nans with majority case 'S'
        data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'

        # Filling nan for Fare and Age based on Pclass and Title medians respectively
        # Age based on Title
        for title in data['Title'].unique():
            data.loc[((data['Title'] == title) &
                       (data['Age'].isnull())), 'Age'] = round(data[data['Title'] == title]['Age'].median())

        # Fare based on Pclass
        for pclass in data['Pclass'].unique():
            data.loc[((data['Pclass'] == pclass) &
                       (data['Fare'].isnull())), 'Fare'] = round(data[data['Pclass'] == pclass]['Fare'].median())

        return Titanic(data)

    def add_groups(self):

        data = self.total
        data['Ticket'] = data['Ticket'].apply(lambda x: x[:-1] + 'X')
        data['Group'] = data.apply(lambda x: '-'.join([x['Surname'], x['Ticket'], str(x['Fare'])]), axis=1)

        return Titanic(data)

    def data_prep_wcg(self):

        data = self.total

        data['Ticket'] = data['Ticket'].apply(lambda x: x[:-1] + 'X')
        data['Group'] = data.apply(lambda x: '-'.join([x['Surname'], x['Ticket'], str(x['Fare'])]), axis=1)
        data['Children'] = data['Age'].apply(lambda x: int(x < 16))
        data['Women'] = data['Sex'].apply(lambda x: int(x == 'female'))
        data['WCG']=0
        data['People'] = 1
        temp = data[['Group', 'Women', 'Children', 'People', 'Survived']].groupby(['Group']).sum()
        groups = temp[temp['People'] > 1].index.unique()
        features = ['WomenSurv','WomenDead','ChildSurv','ChildDead']

        for f in features:
            data[f]=0

        for group in groups:
            data.loc[data['Group']==group,'WCG']=1
            temp = data[data['Group'] == group]
            data.loc[data['Group'] == group, 'WomenSurv'] = temp.apply(
                lambda x: int((x['Sex'] == 'female') & (x['Survived'] == 1)), axis=1)
            data.loc[data['Group'] == group, 'WomenDead'] = temp.apply(
                lambda x: int((x['Sex'] == 'female') & (x['Survived'] == 0)), axis=1)
            data.loc[data['Group'] == group, 'ChildSurv'] = temp.apply(
                lambda x: int((x['Age'] < 16) & (x['Survived'] == 1)), axis=1)
            data.loc[data['Group'] == group, 'ChildDead'] = temp.apply(
                lambda x: int((x['Age'] < 16) & (x['Survived'] == 0)), axis=1)

        # Cabin has too many nans, drop it
        data.drop(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch',
                    'SibSp', 'Ticket', 'Title', 'Surname', 'Group', 'People', 'Sex'], axis=1, inplace=True)

        # One-hot labeling
        # # Label sex
        # total.loc[total['Sex'] == 'male', 'Sex'] = 1
        # total.loc[total['Sex'] == 'female', 'Sex'] = 0

        self.total = data

    def data_model(self):
        x_train, y_train, x_test = self.train, self.labels, self.test
        return x_train, y_train, x_test

    def data_features(self, features_eng):

        if features_eng == 'sex':

            data = self.total
            data = data[['Sex','Survived']]
            data.loc[data['Sex'] == 'male', 'Sex'] = 1
            data.loc[data['Sex'] == 'female', 'Sex'] = 0
            self.total = data

        else:
            print('No operation performed on features')

        return self

    def xgb(self):

        gbm = xgb.XGBClassifier(
            learning_rate=0.01,
            n_estimators=500,
            max_depth=3,
            min_child_weight=2,
            gamma=0.9,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=-1,
            scale_pos_weight=1).fit(self.train, self.labels)

        self.predictions = gbm.predict(self.test).astype(int)

        return self

    def accuracy_score(self):
        print('Accuracy xgboost: {:.4f}'.
              format(accuracy_score(self.predictions, solution()['Survived'].values)))

##
titanic = Titanic().name_features()
total = titanic.total

##
total['Ticket'] = total['Ticket'].str.replace('[./]*','')
ticket_prefix = total['Ticket'].str.findall(r'^(.*) \d*')
ticket_prefix = ticket_prefix[ticket_prefix.apply(lambda x: len(x)>0)].apply(lambda x: x[0])
total['Ticket_pre'] = ticket_prefix
total['Ticket_n'] = total['Ticket'].str.findall(r'(\d*)$').apply(lambda x: x[0])
total.loc[total['Ticket_pre'].isnull(),'Ticket_pre']='STANDARD'
print(ticket_prefix)

##
grouped = total.groupby('Ticket_pre')
total2 = grouped.filter(lambda x: x.Ticket.count()<30)

ticket_pre_sorted = grouped['Ticket'].count().sort_values(ascending=False).keys()
##
for ticket_pre in ticket_pre_sorted:
    print(total.loc[total['Ticket_pre']==ticket_pre,['Name','Age','Parch','SibSp','Cabin','Ticket','Survived']], '\n')
##