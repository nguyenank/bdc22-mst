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

conf_path = os.getcwd()

sys.path.append(conf_path)

from prepare_data import prepare_data

#display stuff
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)

game_df = prepare_data(game_df=pd.read_csv("all_powerplays_4-23-22_cleaned.csv"))

game_df