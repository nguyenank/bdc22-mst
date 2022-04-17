from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sb
import copy
from sklearn.feature_selection import SelectKBest,chi2

#23 of 35 powerplays included
df = pd.read_csv('C:/Users/carli/Documents/Hockey Research/BDC/2022/bdc22-mst/all_powerplays_4-13-22_danger_situations_ozone.csv')
df[["Grid Position","All MST","All_Avg_Edge","All_Total_Edge","All_Avg_Edges per Player","O MST", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player","D MST", "D_Avg_Edge","D_Total_Edge","D_Avg_Edges per Player", "OD_MST_Ratio", "All_OCR"]] = None

pd.set_option('precision', 5)
#TODO make it so that the first 7 coords are the offensive coordinates and the last 7 are the defensive coordinates
#or make sure the offensive team is always on one side of the ice
#or at least change it so that away is on one side and home is on the other side
x_cols = ['away_x_1', 'away_x_2','away_x_3', 'away_x_4','away_x_5', 'away_x_6','away_x_7','home_x_1', 'home_x_2','home_x_3', 'home_x_4','home_x_5', 'home_x_6','home_x_7']
y_cols = ['away_y_1', 'away_y_2','away_y_3', 'away_y_4','away_y_5', 'away_y_6','away_y_7','home_y_1', 'home_y_2','home_y_3', 'home_y_4','home_y_5', 'home_y_6','home_y_7']
positions = ['away_position_1', 'away_position_2','away_position_3', 'away_position_4','away_position_5', 'away_position_6','away_position_7','home_position_1', 'home_position_2','home_position_3', 'home_position_4','home_position_5', 'home_position_6','home_position_7']

#flipping second period positions to match the 1st and 3rd period sides of the ice
df.loc[(df['period'] == 2), 'y_coord'] = (-(df.loc[(df['period'] == 2), 'y_coord']-85/2)+85/2)
df.loc[(df['period'] == 2), y_cols] = (-(df.loc[(df['period'] == 2), y_cols]-85/2)+85/2)
df.loc[(df['period'] == 2), 'x_coord'] = (-(df.loc[(df['period'] == 2), 'x_coord']-100/2)+100/2)
df.loc[(df['period'] == 2), x_cols] = (-(df.loc[(df['period'] == 2), x_cols]-100/2)+100/2)
plt.scatter(df['x_coord'],df['y_coord'])

player_control_area = 5
bins = np.mgrid[0:201:5,0:86:5].T.flatten()
grid = bins.reshape(int(len(bins)/2),2)
for bin in range(len(grid)-1):
    print(grid[bin][0],grid[bin][1])
    print(bin)
    df.loc[(((df['x_coord'] < grid[bin][0]+player_control_area) & (df['x_coord'] >= grid[bin][0])) & ((df['y_coord'] < grid[bin][1]+player_control_area) & (df['y_coord'] >= grid[bin][1]))),"Grid Position"] = bin


def mst_properties(player_positions, player_teams=None):
    #print(player_positions)
    #TODO: players not in the offensive end
    #TODO consider if edge is going to teammate or opponent
    no_players = len(player_positions)
    p2p_distances = np.empty(shape=(no_players, no_players))
    coordinates = player_positions.T
    for i in range(no_players):
        p2p_distances[i] = np.sqrt((coordinates[0]-player_positions[i][0])**2 + (coordinates[1]-player_positions[i][1])**2)
    #print(csr_matrix(p2p_distances))
    tree = minimum_spanning_tree(csr_matrix(p2p_distances)).toarray()
    avg_edge_length = np.mean(tree[~(tree == 0)])
    tot_edge_length = np.sum(tree[~(tree == 0)])
    #print(tree)
    plotted = np.argwhere(tree > 0)
    edge_player_connections = []
    for i in plotted:
        j,k = i
        edge_player_connections.append(j)
        edge_player_connections.append(k)
    unique_vals, edge_player_connections = np.unique(np.array(edge_player_connections),return_counts=True)
    #print(edge_player_connections)
    avg_edges_per_player = np.mean(edge_player_connections)

    #print(avg_edges_per_player)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(coordinates[0],coordinates[1], c = 'khaki', edgecolors='black')
    #no_of_players = len(coordinates[0])
    #ax.set_title("Minimum Spanning Tree")
    #for i in plotted:
    #    j,k = i
    #    #print(j,k)
    #    ax.plot([player_positions[j,0], player_positions[k,0]],[player_positions[j,1],player_positions[k,1]], c = 'dimgray')

    if player_teams is not None:
        pairing_list = []
        for i in plotted:
            j,k = i
            if player_teams[j] == player_teams[k]:
                pairing_list.append(0)
            else:
                pairing_list.append(1)
        print(pairing_list)
        #
        opponent_connection_ratio = np.mean(np.array(pairing_list))

        return tree, avg_edge_length, tot_edge_length, avg_edges_per_player, opponent_connection_ratio

    else:
        return tree, avg_edge_length, tot_edge_length, avg_edges_per_player
#df[x_cols].iloc[28][:7d].count()
#print(df.iloc[785])

#MST variable calculations
for i in range(len(df)):
    x_coords = df[x_cols].iloc[i]
    y_coords = df[y_cols].iloc[i]
    event_team_strength = re.search('(\w) on (\w)',df['situation_type'].iloc[i]).group(1)
    opp_team_strength = re.search('(\w) on (\w)',df['situation_type'].iloc[i]).group(2)
    if ~((x_coords[:7].isnull().all()) | (x_coords[7:].isnull().all())) & (x_coords[:7].count() > 1) & (x_coords[7:].count() > 1):
        print(i)
        raw_coord_pairs = np.array([x_coords,y_coords]).T
        player_teams = np.empty(len(raw_coord_pairs))
        player_teams[:7].fill(0) #one is away
        player_teams[7:].fill(1) #two is home
        player_role = df[positions].iloc[i]
        #print(raw_coord_pairs)
        #print(player_role)
        if "Goalie" in df[positions].iloc[i].unique():
            #print("Goalie")
            raw_coord_pairs = raw_coord_pairs[~(player_role == "Goalie")]
            #print(raw_coord_pairs)
            player_teams = player_teams[~(player_role == "Goalie")]
            #print(raw_coord_pairs)
        all_coord_pairs = raw_coord_pairs[~np.isnan(raw_coord_pairs)]
        player_teams = player_teams[~(np.isnan(raw_coord_pairs)[:,0])]
        all_coord_pairs = all_coord_pairs.reshape(int(len(all_coord_pairs)/2),2)
        plt.scatter(all_coord_pairs.T[0],all_coord_pairs.T[1])
        df['All MST'].iloc[i],df['All_Avg_Edge'].iloc[i], df['All_Total_Edge'].iloc[i], df["All_Avg_Edges per Player"].iloc[i], df["All_OCR"].iloc[i] = mst_properties(all_coord_pairs, player_teams)
        print(df['All_OCR'].iloc[i])


        if df['venue'].iloc[i] == 'home': #home team is the offensive team
            home_coord_pairs = raw_coord_pairs[7:][~np.isnan(raw_coord_pairs[7:])]
            #print(home_coord_pairs)
            home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
            #plt.scatter(home_coord_pairs.T[0],home_coord_pairs.T[1])
            df['O MST'].iloc[i],df['O_Avg_Edge'].iloc[i], df['O_Total_Edge'].iloc[i], df["O_Avg_Edges_per_Player"].iloc[i] = mst_properties(home_coord_pairs)
            #print(df['All MST'].iloc[i])
            away_coord_pairs = raw_coord_pairs[:7][~np.isnan(raw_coord_pairs[:7])]
            #print(away_coord_pairs)
            away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
            #plt.scatter(away_coord_pairs.T[0],away_coord_pairs.T[1])
            df['D MST'].iloc[i],df['D_Avg_Edge'].iloc[i], df['D_Total_Edge'].iloc[i], df["D_Avg_Edges per Player"].iloc[i] = mst_properties(away_coord_pairs)
            #print(df['Away Total Edge'].iloc[i])
        elif df['venue'].iloc[i] == 'away': #away is offensive team
            home_coord_pairs = raw_coord_pairs[7:][~np.isnan(raw_coord_pairs[7:])]
            #print(home_coord_pairs)
            home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
            #plt.scatter(home_coord_pairs.T[0],home_coord_pairs.T[1])
            df['D MST'].iloc[i],df['D_Avg_Edge'].iloc[i], df['D_Total_Edge'].iloc[i], df["D_Avg_Edges per Player"].iloc[i] = mst_properties(home_coord_pairs)
            #print(df['All MST'].iloc[i])
            away_coord_pairs = raw_coord_pairs[:7][~np.isnan(raw_coord_pairs[:7])]
            #print(away_coord_pairs)
            away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
            #plt.scatter(away_coord_pairs.T[0],away_coord_pairs.T[1])
            df['O MST'].iloc[i],df['O_Avg_Edge'].iloc[i], df['O_Total_Edge'].iloc[i], df["O_Avg_Edges_per_Player"].iloc[i] = mst_properties(away_coord_pairs)
            #print("Avg",df["All_Avg_Edges per Player"].iloc[i])
            #print(df['Away Total Edge'].iloc[i])
        #print("Avg one back",df["All_Avg_Edges per Player"].iloc[i-1])
        df["OD_MST_Ratio"].iloc[i] = df['O_Avg_Edge'].iloc[i]/df['D_Avg_Edge'].iloc[i]
        #print(df["OD_MST_Ratio"].iloc[i])
        #if i > 50:
        #    break
        #break

#model development
vars = ["high_danger_within_four","distance_to_attacking_net", "All_Avg_Edge", "All_Total_Edge","O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge","D_Total_Edge","OD_MST_Ratio", "All_OCR"]
ind_vars = copy.deepcopy(vars) #["distance_to_attacking_net","All_Avg_Edge", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge", "D_Total_Edge", "OD_MST_Ratio", "All_OCR"]
ind_vars.remove("high_danger_within_four")
#unused "All_Avg_Edges per Player" "D_Avg_Edges per Player" because extrememly high VIF

df_no_na = df[vars].fillna(value=np.nan).dropna()
#print(df)
#print(df_no_na)
X = df_no_na[ind_vars].reset_index().drop(columns='index')
print(np.shape(X))
pd.set_option("max_rows", None)

#corr_m = X.corr(method ='pearson')
#plt.figure(figsize=(40,40))
#hm = sb.heatmap(corr_m, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 25}, cmap='coolwarm', yticklabels=ind_vars, xticklabels=ind_vars)
#plt.tight_layout()

y = df_no_na['high_danger_within_four']
uniques, counts = np.unique(y, return_counts=True)
scaler = counts[0]/counts[1]
interactions = PolynomialFeatures(interaction_only=True,include_bias = False)
X_w_inter = interactions.fit_transform(X)
new_ind_vars = interactions.get_feature_names_out()
print(new_ind_vars)
#variable selection
selection = SelectKBest(chi2, k=40)
trans_X = selection.fit_transform(X_w_inter, y) #k should be somewhere between 40 and 60
selected_features = new_ind_vars[selection.get_support()]

print(np.shape(np.array(X)))
X_train, X_test, y_train, y_test = train_test_split(trans_X, y, test_size=0.2, random_state=234)

#applying logistic model to training data
model1_log = linear_model.LogisticRegression(solver='liblinear',max_iter=1000, class_weight = {0:1, 1:2.5})
model1_log.fit(X_train, y_train)

#saving features for An
features_coef = pd.DataFrame(np.array([selected_features, model1_log.coef_[0]]).T)
features_coef.to_csv('2022/bdc22-mst/features_coefficients.csv', header=False, index=False)

#model evalution
pred1 = model1_log.predict(X_test)
mse = mean_squared_error(y_test, pred1)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("Log Score: ", model1_log.score(X_test, y_test))
print(confusion_matrix(y_test, model1_log.predict(X_test)))

pred_train = model1_log.predict(X_train)
mse = mean_squared_error(y_train, pred_train)
print("The mean squared error (MSE) on train set: {:.4f}".format(mse))
print("Log Score: ", model1_log.score(X_train, y_train))
print(confusion_matrix(y_train, model1_log.predict(X_train)))
sb.kdeplot(model1_log.predict_proba(X_test)[:,1], cut=0)
sb.kdeplot(model1_log.predict_proba(X_test)[:,0], cut=0)
#sb.kdeplot(model1_log.predict_proba(X_test)[:,1])

#Import roc_curve, auc
from sklearn.metrics import roc_curve, auc
#Calculate the probability scores of each point in the training set
y_train_score = model1_log.decision_function(X_train)
# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
#Calculate the probability scores of each point in the test set
y_test_score = model1_log.decision_function(X_test)
#Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)

plt.plot(test_fpr, test_tpr, color='darkorange', lw=1, label='ROC curve')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("AUC", auc(test_fpr,test_tpr))

from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(y_test, y_test_score, name="Log")
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.plot([1, 0], [0, 1],'r--')
plt.show()


#random forest and xGb overfit

#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.values.shape[1])]
#vif["features"] = X_train.columns
#print(vif.round(1))
