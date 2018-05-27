import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('buttle_data.csv')
test_data = pd.read_csv('buttle_data.csv')

data = data.drop('Unnamed: 0', axis=1)
data.head()

data_train, data_test = train_test_split(data)

X_train = data_train.drop(['Win?'], axis=1)
Y_train = data_train['Win?']
X_test = data_test.drop(['Win?'], axis=1)
Y_test = data_test['Win?']

clf = LogisticRegression()
clf.fit(X_train, Y_train)
predicted = pd.DataFrame({'LogisPredicted':clf.predict(X_test)})

clf.score(X_test, Y_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 0〜1.0

X_test_value = clf.decision_function(X_test)
X_test_prob = sigmoid(X_test_value)
actual = pd.Series(X_test_prob).round(5)
predicted['actual'] = actual.values.tolist()


def logloss(predicted):
    sum = 0
    for i in predicted.index:
        sum += (predicted.iat[i,0] * math.log(predicted.iat[i,1]) + (1-predicted.iat[i,0]) * math.log(1 - predicted.iat[i,1]))
    return - sum / len(predicted.index)

print(logloss(predicted))
