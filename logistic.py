import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('buttle_data.csv')
test_data = pd.read_csv('buttle_data.csv')

## 変数
c = 1000
n = 1


data = data.drop('Unnamed: 0', axis=1)

data_train, data_test = train_test_split(data, test_size=0.3, random_state=0)

X_train = data_train.drop(['Win?'], axis=1)
Y_train = data_train['Win?']
X_test = data_test.drop(['Win?'], axis=1)
Y_test = data_test['Win?']

clf = LogisticRegression(C=c)
clf.fit(X_train, Y_train)
predicted = pd.DataFrame({'expect':Y_test})

# print(clf.score(X_test, Y_test))

def sigmoid(x):
    return 1 / (1 + np.exp(- (n * x))) # 0〜1.0

X_test_value = clf.decision_function(X_test)
X_test_prob = sigmoid(X_test_value)
actual = pd.Series(X_test_prob)
predicted['probability'] = actual.values.tolist()


def logloss(predicted):
    sum = 0
    for i in range(len(predicted.index)):
        y_prediction = predicted.iat[i,1]
        sum += (predicted.iat[i,0] * math.log(y_prediction) + (1-predicted.iat[i,0]) * math.log(1 - y_prediction))
    return - sum / len(predicted.index)

print(logloss(predicted))

f = open('result.txt', 'w') # 書き込みモードで開く
f.write(str(logloss(predicted))) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる



####--------------
# ここからテスト結果出力用
comp_test_data = pd.read_csv('buttle_data_test.csv')

comp_test_data = comp_test_data.drop('Unnamed: 0', axis=1)
# print(comp_test_data.head())
# result = pd.DataFrame(comp_test_data.index, columns=['id'])


comp_test_value = clf.decision_function(comp_test_data)
comp_test_prob = sigmoid(comp_test_value)
probability = pd.Series(comp_test_prob).round(6)
result = pd.DataFrame(probability.values.tolist(), columns=['probability'])
result.index.name = "id"
result.to_csv('submission.csv', float_format='%.6f')
