import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix,roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sb
import copy
from sklearn.feature_selection import SelectKBest,chi2
from hockey_rink import BDCRink
import hockey_mst
from sklearn.metrics import PrecisionRecallDisplay

#display stuff
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)


## Residual Data Cleaning
unflipped_df = pd.read_csv('C:/Users/carli/Documents/Hockey Research/BDC/2022/bdc22-mst/all_powerplays_4-13-22_danger_situations_ozone.csv') #24 of 35 powerplays included
unflipped_df[["O Players","D Players","All MST","All_Avg_Edge","All_Total_Edge","All_Avg_Edges per Player","O MST", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player","D MST", "D_Avg_Edge","D_Total_Edge","D_Avg_Edges per Player", "OD_MST_Ratio", "All_OCR"]] = None

#finding columns of x and y coordinates as well as positions
x_cols = ['away_x_1', 'away_x_2','away_x_3', 'away_x_4','away_x_5', 'away_x_6','away_x_7','home_x_1', 'home_x_2','home_x_3', 'home_x_4','home_x_5', 'home_x_6','home_x_7']
y_cols = ['away_y_1', 'away_y_2','away_y_3', 'away_y_4','away_y_5', 'away_y_6','away_y_7','home_y_1', 'home_y_2','home_y_3', 'home_y_4','home_y_5', 'home_y_6','home_y_7']
positions = ['away_position_1', 'away_position_2','away_position_3', 'away_position_4','away_position_5', 'away_position_6','away_position_7','home_position_1', 'home_position_2','home_position_3', 'home_position_4','home_position_5', 'home_position_6','home_position_7']

#flipping second period positions to match the 1st and 3rd period sides of the ice
df = copy.deepcopy(unflipped_df)
print(unflipped_df[['x_coord','away_x_1', 'away_x_2','away_x_3', 'away_x_4','away_x_5', 'away_x_6','away_x_7','home_x_1', 'home_x_2','home_x_3', 'home_x_4','home_x_5', 'home_x_6','home_x_7']].iloc[358])
df.loc[(df['period'] == 2), 'y_coord'] = (-(unflipped_df.loc[(df['period'] == 2), 'y_coord']-85/2)+85/2)
df.loc[(df['period'] == 2), y_cols] = (-(unflipped_df.loc[(df['period'] == 2), y_cols]-85/2)+85/2)
df.loc[(df['period'] == 2), 'x_coord'] = (-(unflipped_df.loc[(df['period'] == 2), 'x_coord']-200/2)+200/2)
df.loc[(df['period'] == 2), x_cols] = (-(unflipped_df.loc[(df['period'] == 2), x_cols]-200/2)+200/2)

#extra variable calcuation
df_w_ind_vars = hockey_mst.ind_var_calculation(df, x_cols, y_cols, positions)

##Model Development

vars = ["high_danger_within_four","distance_to_attacking_net", "All_Avg_Edge", "All_Total_Edge","O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge","D_Total_Edge","OD_MST_Ratio", "All_OCR"]
ind_vars = copy.deepcopy(vars) #["distance_to_attacking_net","All_Avg_Edge", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge", "D_Total_Edge", "OD_MST_Ratio", "All_OCR"]
ind_vars.remove("high_danger_within_four")
#unused:
#"All_Avg_Edges per Player" because extrememly high VIF
#"D_Avg_Edges per Player" because extrememly high VIF
## of d players and # of o players because makes it worse :(

#fiddling around with the data to select the relevant dependent and independent variables in a way the relevant extraneous info can still be used when plotting on rinks
df_all_no_na = df_w_ind_vars.fillna(value=np.nan).dropna(subset = vars) #df with all columns but with rows dropped where one of the modelling variables is NaN
df_no_na = df_all_no_na[vars] #keeping only the relevant variables to modelling
X = df_no_na[ind_vars].reset_index().drop(columns='index') #keeping only the relevant independent variables to modelling
y = df_no_na['high_danger_within_four'] #keeping only the relevant dependent variable to modelling

#correlation matrix
#corr_m = X.corr(method ='pearson')
#plt.figure(figsize=(40,40))
#hm = sb.heatmap(corr_m, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 25}, cmap='coolwarm', yticklabels=ind_vars, xticklabels=ind_vars)
#plt.tight_layout()

#get ratio of dep variable classes
uniques, counts = np.unique(y, return_counts=True)
scaler = counts[0]/counts[1]

#setting up interaction terms
interactions = PolynomialFeatures(interaction_only=True,include_bias = False)
X_w_inter = interactions.fit_transform(X)
new_ind_vars = interactions.get_feature_names_out()

#variable selection - honestly not the best variable selection method but it's the one I can do without fucking up the data and retaining variable names to give to An
selection = SelectKBest(chi2, k=40)
trans_X = selection.fit_transform(X_w_inter, y) #k should be somewhere between 40 and 60 otherwise model is :(
selected_feature_names = new_ind_vars[selection.get_support()]

#splitting training and testing
X_train, X_test, y_train, y_test = train_test_split(trans_X, y, test_size=0.2, random_state=366)

#applying logistic model to training data
model1_log = linear_model.LogisticRegression(solver='liblinear',max_iter=10000, class_weight = {0:1, 1:2.5}, random_state=43)
model1_log.fit(X_train,y_train)

#get features selected by SelectKBest and also their coefficients in the model to put in saved json object
features_coef = pd.DataFrame(np.array([selected_features, model1_log.coef_[0]]).T)
#features_coef.to_csv('2022/bdc22-mst/features_coefficients.csv', header=False, index=False)

##Model Evalution
#MSE, Score, and Confusion matrix for train and test data
pred1 = model1_log.predict(X_test)
mse = mean_squared_error(y_test, pred1)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("Logistic Regression Score: ", model1_log.score(X_test, y_test))
print(confusion_matrix(y_test, model1_log.predict(X_test)))

pred_train = model1_log.predict(X_train)
mse = mean_squared_error(y_train, pred_train)
print("The mean squared error (MSE) on train set: {:.4f}".format(mse))
print("Logistic Regression Score: ", model1_log.score(X_train, y_train))
print(confusion_matrix(y_train, model1_log.predict(X_train)))

#honestly not sure what these graphs say but it makes sense to me?
#sb.kdeplot(model1_log.predict_proba(X_test)[:,1], cut=0)
#sb.kdeplot(model1_log.predict_proba(X_test)[:,0], cut=0)
#plt.show()

#ROC and PRC plot prep

#Calculate the probability scores of each point in the training set
y_train_score = model1_log.decision_function(X_train)
# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
#Calculate the probability scores of each point in the test set
y_test_score = model1_log.decision_function(X_test)
#Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)

#ROC Curve
sb.lineplot(test_fpr, test_tpr, color='darkorange', lw=1, label='ROC curve', ci=False)
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


## Plotting, still fiddling with this
for i in [0,358,35,37,282]:
    x_coords = df_all_no_na[x_cols].iloc[i]
    y_coords = df_all_no_na[y_cols].iloc[i]
    raw_coord_pairs = np.array([x_coords,y_coords]).T
    prob = model1_log.predict_proba(np.array([trans_X[i]]))[:,1][0] #the second value in each row output by predict_proba is probability of a 1. check model1_log.classes_ for reference
    print(df_all_no_na.iloc[i])
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    rink = BDCRink()
    rink.draw(ax=ax)
    player_role = df_all_no_na[positions].iloc[i]
    #print(raw_coord_pairs)
    print(player_role)
    if df_all_no_na['venue'].iloc[i] == 'home': #home team is the offensive team
        raw_home_coord_pairs = raw_coord_pairs[7:][~(player_role[7:] == "Goalie")]
        home_coord_pairs = raw_home_coord_pairs[~np.isnan(raw_home_coord_pairs)]
        home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
        print(home_coord_pairs)
        rink.scatter(home_coord_pairs.T[0],home_coord_pairs.T[1], label = "Offensive", c='lightgreen', s=80, ax = ax)
        rink.scatter(df_all_no_na['x_coord'].iloc[i],df_all_no_na['y_coord'].iloc[i], label = "Puck Location", c='black', s=80, ax = ax)
        #print(df['All MST'].iloc[i])
        raw_away_coord_pairs = raw_coord_pairs[:7]#[~(player_role[7:] == "Goalie")]
        away_coord_pairs = raw_away_coord_pairs[~np.isnan(raw_away_coord_pairs)]
        away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
        print(away_coord_pairs)
        rink.scatter(away_coord_pairs.T[0],away_coord_pairs.T[1], label = "Defensive", c='mediumpurple', s=80, ax = ax)
    elif df_all_no_na['venue'].iloc[i] == 'away': #away is offensive team
        raw_home_coord_pairs = raw_coord_pairs[7:]#[~(player_role[7:] == "Goalie")]
        home_coord_pairs = raw_home_coord_pairs[~np.isnan(raw_home_coord_pairs)]
        home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
        rink.scatter(home_coord_pairs.T[0],home_coord_pairs.T[1], label = "Defensive", c='mediumpurple', s=80, ax=ax)
        raw_away_coord_pairs = raw_coord_pairs[:7][~(player_role[7:] == "Goalie")]
        away_coord_pairs = raw_away_coord_pairs[~np.isnan(raw_away_coord_pairs)]
        away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
        print(away_coord_pairs)
        rink.scatter(away_coord_pairs.T[0], away_coord_pairs.T[1], label = "Offensive", c='lightgreen', s=80, ax=ax)
        rink.scatter(df_all_no_na['x_coord'].iloc[i],df_all_no_na['y_coord'].iloc[i], label = "Puck Location", c='black', s=80, ax=ax)
    plt.title("Probability of a Dangerous Situation = "+str(prob), fontsize = 20)
    leg = plt.legend(fontsize = 15)
    leg.set_zorder(120)
    plt.show()
