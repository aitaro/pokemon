import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib
matplotlib.use('Agg')
# matplotlib.use("tkagg")
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import math

# %matplotlib inline

from matplotlib.pylab import rcParams
import seaborn as sns
rcParams['figure.figsize'] = 10, 4

data = pd.read_csv('buttle_data.csv')
data = data.drop('Unnamed: 0', axis=1)

data_train, data_test = train_test_split(data, test_size=0.3, random_state=0)

# X_train = data_train.drop(['Win?'], axis=1)
# Y_train = data_train['Win?']
X_test = data_test.drop(['Win?'], axis=1)
Y_test = data_test['Win?']

train = data_train

target = 'Win?'

def sigmoid(x):
    return 1 / (1 + np.exp(- (1 * x)))

def logloss(predicted):
    sum = 0
    for i in range(len(predicted.index)):
        y_prediction = predicted.iat[i,1]
        sum += (predicted.iat[i,0] * math.log(y_prediction) + (1-predicted.iat[i,0]) * math.log(1 - y_prediction))
    return - sum / len(predicted.index)

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Win?'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Win?'], cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print ("Model Report")
    print ("Accuracy : {:.4f}".format(metrics.accuracy_score(dtrain['Win?'].values, dtrain_predictions)))
    print ("AUC Score (Train): {:.4f}".format(metrics.roc_auc_score(dtrain['Win?'], dtrain_predprob)))

    if performCV:
        print ("CV Score : Mean - {:.6f} | Std - {:.6f} | Min - {:.6f} | Max - {:.6f}".format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        sns.set_palette("husl")
        sns.barplot(feat_imp.head(10).index, feat_imp.head(10).values)
        plt.title('Top10 Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.xticks(rotation=60)
        plt.show()

    #print results
    predicted = pd.DataFrame({'expect':Y_test})
    X_test_value = alg.decision_function(X_test)
    X_test_prob = sigmoid(X_test_value)
    actual = pd.Series(X_test_prob)
    predicted['probability'] = actual.values.tolist()
    print(logloss(predicted))

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target]]
gbm0 = GradientBoostingClassifier(learning_rate=0.2, n_estimators=400, random_state=10)
modelfit(gbm0, train, predictors)


# predictors = [x for x in train.columns if x not in [target]]
# param_test1 = {'n_estimators':list(range(200,601,100))}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
# param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(train[predictors],train[target])
# mean_list = []
# for s in gsearch1.grid_scores_:
#     mean_list.append(s[1])
# print(mean_list)
# gsearch1.grid_scores_, gsearch1.best_params_

from mpl_toolkits.mplot3d import Axes3D
import time
start_time = time.time()
param_test2 = {'max_depth':list(range(9,11,2)), 'min_samples_split':list(range(600,801,200))}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, n_estimators=400, max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
x=[]
y=[]
z=[]
for s in gsearch2.grid_scores_:
    x.append(s[0]["max_depth"])
    y.append(s[0]["min_samples_split"])
    z.append(s[1])
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(x, y, z)
# plt.show()
print('Showing results took {} secs.'.format(time.time() - start_time))
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
