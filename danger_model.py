"""Importing libraries and functions used for modeling purposes"""
# TODO
# try downsampling to improve the ratio of observations rather than upsampling
import json
import copy
from turtle import pen

# Data Manipulation Packages
import pandas as pd 
import numpy as np

# Plotting Packages
import matplotlib.pyplot as plt
import seaborn as sns
from hockey_rink import BDCRink

# Modeling Packages/Functions
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest,
    chi2
)
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc,
    PrecisionRecallDisplay
)

# Custom-built functions
import hockey_mst
# from modelling_and_plotting import X_w_inter # applies the MST calculations to a given dataframe
from prepare_data import prepare_data # function that preps the dataframe for the hockey_mst function
from split_data import split_data # splits data into variables + target
from data_partition import (
    resample_data,
    data_partition
    ) # re-sample to bring 0/1 classes to roughly even (or preferred proportion)

# editing display options
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)

game_df = prepare_data(game_df=pd.read_csv("all_powerplays_4-23-22_cleaned_final.csv"))
# game_df = game_df[game_df.angle_to_attacking_net > 0]
# game_df.high_danger_within_four.value_counts()
game_df = data_partition(game_df=game_df, prop=0.25)
# game_df.high_danger_within_four.value_counts() # fun to compare lol
x, y = split_data(game_df=game_df)

print(len(y[y == 1]), " successes", sep="")
print(len(y[y == 0]), " not successes", sep="")

def get_interactions(x): 
    interactions = PolynomialFeatures(interaction_only=True, include_bias=True)
    x_w_inter = interactions.fit_transform(X=x)
    inter_vars_raw = interactions.get_feature_names_out()

    new_names = []

    for i in inter_vars_raw:
        n = i.replace(' ', '_')
        new_names.append(n)

    return x_w_inter, new_names, inter_vars_raw

x_w_inter, new_names, inter_vars_raw = get_interactions(x = x)
# Carlie, run the code above and this line below and it shows that there are negative values
# pd.set_option('display.max_rows', 600)
# (x_w_inter < 0).any()
# x_df = pd.DataFrame(x_w_inter, columns=new_names)
# (x_df < 0).any(0)

# x_df.loc[:, x_df.columns.str.endswith('attacking_net')]
# (x_w_inter < 0).any(0)

def variable_selection(x_w_inter, y, new_names, k = 40):
    new_names = np.array(new_names)

    selection = SelectKBest(chi2, k = k)
    trans_x = selection.fit_transform(x_w_inter, y)
    raw_selected_names = new_names[selection.get_support()]

    raw_names = []

    for i in raw_selected_names:
        raw_names.append(i)

    return trans_x, raw_names

trans_x, raw_names = variable_selection(x_w_inter=x_w_inter, y = y, new_names=new_names)

x_train, x_test, y_train, y_test = train_test_split(trans_x, y, test_size=0.2, random_state=366)

test = pd.DataFrame(x_test).join(y_test.reset_index(drop = True)).drop_duplicates()
x_test = np.array(test.drop(columns=['high_danger_within_four']))
y_test = test['high_danger_within_four']
# x_test = np.array(pd.DataFrame(x_test).drop_duplicates())
# y_test.reset_index(drop = True)

model1_log = linear_model.LogisticRegression(solver='liblinear', max_iter=10000, class_weight={0:1, 1:2}, penalty='l1', random_state=43)
model1_log.fit(x_train, y_train)

raw_names.append('intercept')
feature_dict = {
    key: None for key in raw_names
}
model1_log.coef_[0]
print(feature_dict)
m = 0
for feature in feature_dict:
    if feature == 'intercept':
        feature_dict['intercept'] = model1_log.intercept_[0]
    else:
        feature_dict[feature] = model1_log.coef_[0][m]
    m += 1
with open("feature_dict.json", "w") as outfile:
    json.dump(feature_dict, outfile)

pred1 = model1_log.predict(x_test)
mse = mean_squared_error(y_test, pred1)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("Logistic Regression Score: ", model1_log.score(x_test, y_test))
print(confusion_matrix(y_test, model1_log.predict(x_test)))

pred_train = model1_log.predict(x_train)
mse = mean_squared_error(y_train, pred_train)
print("The mean squared error (MSE) on train set: {:.4f}".format(mse))
print("Logistic Regression Score: ", model1_log.score(x_train, y_train))
print(confusion_matrix(y_train, model1_log.predict(x_train)))

#Calculate the probability scores of each point in the training set
y_train_score = model1_log.decision_function(x_train)
# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
#Calculate the probability scores of each point in the test set
y_test_score = model1_log.decision_function(x_test)
#Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)

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
display = PrecisionRecallDisplay.from_predictions(y_test, y_test_score, name="Log")
plt.title("Testing Set 2-class Precision-Recall curve")
plt.plot([1, 0], [0, 1],'r--')
plt.show()

display = PrecisionRecallDisplay.from_predictions(y_train, y_train_score, name="Log")
plt.title("Training Set 2-class Precision-Recall curve")
plt.plot([1, 0], [0, 1],'r--')
plt.show()