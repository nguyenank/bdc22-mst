from copy import copy
from multiprocessing import dummy
from pickletools import read_bytes1
from turtle import pos
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV
)
from sklearn.feature_selection import (
    SelectKBest,
    chi2
)
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, scale
import copy
import pandas as pd
import os
import csv
import sys
from hockey_rink import BDCRink
import random
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier
# from modelling_and_plotting import X_w_inter

conf_path = os.getcwd()

sys.path.append(conf_path)

from prepare_data import prepare_data
from split_data import split_data

plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)
# game_df=pd.read_csv("all_powerplays_4-23-22_cleaned.csv")
game_df = prepare_data(game_df=pd.read_csv("all_powerplays_4-23-22_cleaned.csv"))

x, y = split_data(game_df=game_df)

game_df.high_danger_within_four.value_counts()

interations = PolynomialFeatures(interaction_only=True, include_bias=True)
x_w_inter = interations.fit_transform(x)
inter_vars_raw = interations.get_feature_names_out()

new_names = []

for i in inter_vars_raw:
    n = i.replace(' ', '_')
    new_names.append(n)

param = [{
    'C': [10**-2,10**-1,10**0,10**1,10**2,10**3], 
    'penalty': ['l1'], 
    'tol': [10**-5, 10**-4, 10**-3]
}]
# C = 1/10, tol = 1**-5
train_x, train_y, test_x, test_y = train_test_split(x_w_inter, y, test_size=0.25, random_state=366)

test_x = test_x.astype('int64')
test_y = test_y.astype('int64')

# TODO
# - re-sampling for more positive cases
# - class weighing too
# we prefer to have false positives bc there are def high danger situations
# that don't result in a high danger shot but are still high danger (carlie explains it well)

# sm = SMOTE(random_state=123)

# train_x, test_x = sm.fit_resample(train_x, test_x.ravel())

# lr_model = LogisticRegression(solver='liblinear', max_iter=10000, random_state=123)
# gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=10, verbose=4)

# gs_model.fit(train_x, test_x)

model1_log = LogisticRegression(solver='liblinear', 
                                max_iter=10000, 
                                random_state=123,
                                C=1, tol=10**-5)
model1_log.fit(train_x, test_x)

#applying logistic model to training data
# model1_log = LogisticRegressionCV(solver='liblinear', penalty="l2", max_iter=10000, random_state=43)
# model1_log.fit(train_x, test_x)

# TruePos FalsePos
# FalseNeg TrueNeg

dummy_clf = DummyClassifier(random_state=123, strategy='stratified')
dummy_clf.fit(train_x, test_x)
dpred = dummy_clf.predict(train_y)

mean_squared_error(test_y, dpred)
print("Logistic Regression Score for dummy: ", dummy_clf.score(train_y, test_y))
d = pd.DataFrame(test_y.reset_index(drop = True)).join(pd.DataFrame(dpred, columns=['pred']))
confusion_matrix(d.high_danger_within_four, d.pred)

pred = model1_log.predict(train_y)
mean_squared_error(test_y, pred)
print("Logistic Regression Score for test: ", model1_log.score(train_y, test_y))
re1 = pd.DataFrame(test_y.reset_index(drop = True)).join(pd.DataFrame(pred, columns=['pred']))
confusion_matrix(re1.high_danger_within_four, re1.pred)

tpred = model1_log.predict(train_x)
mean_squared_error(test_x, tpred)
print("Logistic Regression Score for training: ", model1_log.score(train_x, test_x))
res = pd.DataFrame(test_x.reset_index(drop = True)).join(pd.DataFrame(tpred, columns=['pred']))
confusion_matrix(res.high_danger_within_four, res.pred)

print("Logistic Regression Score for test: ", 
        round(model1_log.score(train_y, test_y), 4), 
    "\nConfusion Matrix for Test Set:\n", 
        confusion_matrix(re1.high_danger_within_four, re1.pred), 
    "\nLogistic Regression Score for training: ", 
        round(model1_log.score(train_x, test_x), 4),
    "\nConfusion Matrix for Training Set:\n", 
        confusion_matrix(res.high_danger_within_four, res.pred),
    "\nLogistic Regression Score for Dummy: ",
        round(dummy_clf.score(train_y, test_y), 4),
    "\nConfusion Matrix for Dummy:\n",
        confusion_matrix(d.high_danger_within_four, d.pred))

# 0.85 on test
# 0.90 in train

y_train_score = model1_log.decision_function(train_x)
train_fpr, train_tpr, thresholds = roc_curve(test_x, y_train_score)

y_test_score = model1_log.decision_function(train_y)
#Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(test_y, y_test_score)

#ROC Curve
sns.lineplot(test_fpr, test_tpr, color='darkorange', lw=1, label='ROC curve', ci=False)
plt.legend(loc = 'lower right')
plt.title("AUC "+str(auc(test_fpr,test_tpr)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Precision Recall Curves for testing and training sets
display = PrecisionRecallDisplay.from_predictions(test_y, y_test_score, name="Log")
plt.title("Testing Set 2-class Precision-Recall curve")
plt.plot([1, 0], [0, 1],'r--')
plt.show()

display = PrecisionRecallDisplay.from_predictions(test_x, y_train_score, name="Log")
plt.title("Training Set 2-class Precision-Recall curve")
plt.plot([1, 0], [0, 1],'r--')
plt.show()