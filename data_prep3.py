from titanic import *

titanic = Titanic()

total = titanic.total

# Family feature engineering
total['Family'] = total['SibSp'] + total['Parch'] + 1
total['isAlone'] = 0
total.loc[total['Family'] == 1, 'isAlone'] = 1
total.drop(['SibSp', 'Parch'], axis=1, inplace=True)

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


total['Group'] = 0
group_label = 0
for surname in total['Surname'].unique():
    same_name = total[((total['Surname'] == surname) & (total['Group'] == 0) & (total['isAlone'] == 0))]
    group_label = group_label + 1

    for ticket in same_name['Ticket'].unique():
        total.loc[(((total['Surname'] == surname) | (total['Ticket'].apply(lambda x: x[:-2]) == ticket[:-2]))
                   & (total['isAlone'] == 0)), 'Group'] = group_label
total['Group'] = total['Group'].apply(lambda x: str(x))
total['Pclass'] = total['Group'].apply(lambda x: str(x))

fare_intervals = [(0, 10), (10, 25), (25, 50), (50, 1000)]
total = survival_rate_feature(total, 'Fare', 'FareRate', fare_intervals)

# 5 fasce di eta'
age_intervals = [(0, 1), (1, 10), (10, 25), (25, 50), (60, 100)]
total = survival_rate_feature(total, 'Age', 'AgeRate', age_intervals)

# Label sex
total.loc[total['Sex'] == 'male', 'Sex'] = 1
total.loc[total['Sex'] == 'female', 'Sex'] = 0

# Cabin has too many nans, drop it
total.drop(['Cabin', 'Ticket', 'PassengerId', 'Surname'], axis=1, inplace=True)

#
# Labels
total = pd.get_dummies(total)

train_set = total[total['train'] == 1]
test_set = total[total['train'] == 0]

y_train = train_set['Survived']
x_train = train_set.drop(['Survived', 'train'], axis=1)
x_test = test_set.drop(['Survived', 'train'], axis=1)
