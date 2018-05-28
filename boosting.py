import numpy as np
import pandas as pd
import math
import gbdtree as gb
import matplotlib.pyplot as plt




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('buttle_data.csv')
test_data = pd.read_csv('buttle_data.csv')


data = data.drop('Unnamed: 0', axis=1)

data_train, data_test = train_test_split(data, test_size=0.3, random_state=0)

X_train = data_train.drop(['Win?'], axis=1)
Y_train = data_train['Win?']
X_test = data_test.drop(['Win?'], axis=1)
Y_test = data_test['Win?']
# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                             random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Y_train)

print(clf.score(X_test, Y_test))
print(clf.predict(X_test))
predicted = pd.DataFrame({'expect':Y_test})
predicted['probability'] = actual.values.tolist()
