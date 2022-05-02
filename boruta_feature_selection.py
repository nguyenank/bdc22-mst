from copy import copy
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
# from modelling_and_plotting import X_w_inter

conf_path = os.getcwd()

sys.path.append(conf_path)

from prepare_data import prepare_data
from split_data import split_data

#display stuff
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)

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
    'tol': [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
}]
lr_model = LogisticRegression(solver='liblinear', max_iter=10000, random_state=123)
gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=3)

# https://datascience.stackexchange.com/questions/93861/trouble-performing-feature-selection-using-boruta-and-support-vector-regression
# helpful

# permutation feature importance?
# https://christophm.github.io/interpretable-ml-book/feature-importance.html

lr_model.fit(x_w_inter, y)

# feat_selector = BorutaPy(estimator=lr_model, n_estimators='auto', 
#                          verbose=2, random_state=123)

# feat_selector.fit(np.array(x_w_inter), np.array(y))