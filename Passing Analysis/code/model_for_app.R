# Load (install if needed) the required packages
load.libraries = c("readr","tidyverse","dplyr","randomForest","xgboost")
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for (libs in install.lib) {install.packages(libs, dependencies = TRUE)}
sapply(load.libraries, require, character = TRUE)

all_data = read_csv("/Volumes/BRICK_HOUSE/Hockey/Current Git/Big-Data-Cup-2022-Private/all_data.csv")

data_4_model = all_data %>% 
  mutate(goal_diff = goals_for-goals_against) %>% 
  mutate(man_adv = as.numeric(substr(situation_type,0,1))-as.numeric(substr(situation_type,6,7))) %>%
  select(team_name,opp_team_name,venue,period,
         distance_to_attacking_net, angle_to_attacking_net,Home_Plate_Control,Rink_Control,
         Max_Success,Max_Best,Max_Exp,Passer_Value,Max_Player_Success, Max_Player_Best, Max_Player_Exp,
         Max_Player_Success_X,Max_Player_Success_Y,Max_Player_Best_X,Max_Player_Best_Y,Max_Player_Exp_X,Max_Player_Exp_Y,
         Mean_Player_Success, Mean_Player_Best, Mean_Player_Exp,goal_diff,man_adv,assumed_danger_states) 
data_4_model = data_4_model[!is.na(data_4_model$Home_Plate_Control),]

data_4_model$split = 0
data_4_model$split[which(data_4_model$assumed_danger_states==0)] = ifelse(runif(n=length(which(data_4_model$assumed_danger_states==0)))>0.7, yes=1, no=0)
data_4_model$split[which(data_4_model$assumed_danger_states==1)] = ifelse(runif(n=length(which(data_4_model$assumed_danger_states==1)))>0.7, yes=1, no=0)

y_train = data_4_model$assumed_danger_states[data_4_model$split==0] %>% as.factor()
y_test = data_4_model$assumed_danger_states[data_4_model$split==1] %>% as.factor()
x_train = data_4_model[data_4_model$split==0,1:20]
x_test = data_4_model[data_4_model$split==1,1:20]

#Random Forest
p = ncol(x_train)
rf1 <- randomForest(x=x_train, y=y_train, importance=TRUE, ntree=2500,
                          mtry=1, keep.forest=TRUE)
rf1$confusion
varImpPlot(rf1)
test_preds_rf1 = predict(rf1,newdata=x_test)
conf_rf1 = mmetric(y_test,test_preds_rf1,metric="CONF")$conf
#sensitivity
conf_rf1[1,1]/sum(conf_rf1[1,])
#precision
conf_rf1[1,1]/sum(conf_rf1[,1])
#specificity
conf_rf1[2,2]/sum(conf_rf1[2,])
#accuracy
(conf_rf1[1,1]+conf_rf1[2,2])/sum(conf_rf1)

rf2 <- randomForest(x=x_train, y=y_train, importance=TRUE, ntree=2500,
                    mtry=round(p/5), keep.forest=TRUE)
rf2$confusion
varImpPlot(rf2)
test_preds_rf2 = predict(rf2,newdata=x_test)
conf_rf2 = mmetric(y_test,test_preds_rf2,metric="CONF")$conf
#sensitivity
conf_rf2[1,1]/sum(conf_rf2[1,])
#precision
conf_rf2[1,1]/sum(conf_rf2[,1])
#specificity
conf_rf2[2,2]/sum(conf_rf2[2,])
#accuracy
(conf_rf2[1,1]+conf_rf2[2,2])/sum(conf_rf2)

rf3 <- randomForest(x=x_train, y=y_train, importance=TRUE, ntree=2500,
                    mtry=round(p/4), keep.forest=TRUE)
rf3$confusion
varImpPlot(rf3)
test_preds_rf3 = predict(rf3,newdata=x_test)
conf_rf3 = mmetric(y_test,test_preds_rf3,metric="CONF")$conf
#sensitivity
conf_rf3[1,1]/sum(conf_rf3[1,])
#precision
conf_rf3[1,1]/sum(conf_rf3[,1])
#specificity
conf_rf3[2,2]/sum(conf_rf3[2,])
#accuracy
(conf_rf3[1,1]+conf_rf3[2,2])/sum(conf_rf3)

rf4 <- randomForest(x=x_train, y=y_train, importance=TRUE, ntree=2500,
                    mtry=round(p/3), keep.forest=TRUE)
rf4$confusion
varImpPlot(rf4)
test_preds_rf4 = predict(rf4,newdata=x_test)
conf_rf4 = mmetric(y_test,test_preds_rf4,metric="CONF")$conf
#sensitivity
conf_rf4[1,1]/sum(conf_rf4[1,])
#precision
conf_rf4[1,1]/sum(conf_rf4[,1])
#specificity
conf_rf4[2,2]/sum(conf_rf4[2,])
#accuracy
(conf_rf4[1,1]+conf_rf4[2,2])/sum(conf_rf4)

rf5 <- randomForest(x=x_train, y=y_train, importance=TRUE, ntree=2500,
                    mtry=round(p/2), keep.forest=TRUE)
rf5$confusion
varImpPlot(rf5)
test_preds_rf5 = predict(rf5,newdata=x_test)
conf_rf5 = mmetric(y_test,test_preds_rf5,metric="CONF")$conf
conf_rf5
#sensitivity
conf_rf5[1,1]/sum(conf_rf5[1,])
#precision
conf_rf5[1,1]/sum(conf_rf5[,1])
#specificity
conf_rf5[2,2]/sum(conf_rf5[2,])
#accuracy
(conf_rf5[1,1]+conf_rf5[2,2])/sum(conf_rf5)
#test error
mean(test_preds_rf5 != y_test)


x_train_dummies = data.frame(Canada = ifelse(x_train$team_name=="Olympic (Women) - Canada" , yes=1, no=0),
                              USA = ifelse(x_train$team_name=="Olympic (Women) - United States" , yes=1, no=0),
                              Finland = ifelse(x_train$team_name=="Olympic (Women) - Finland" , yes=1, no=0),
                              Switz = ifelse(x_train$team_name=="Olympic (Women) - Switzerland" , yes=1, no=0),
                              OAR = ifelse(x_train$team_name=="Olympic (Women) - Olympic Athletes from Russia" , yes=1, no=0),
                              Opp_Canada = ifelse(x_train$opp_team_name=="Olympic (Women) - Canada" , yes=1, no=0),
                              Opp_USA = ifelse(x_train$opp_team_name=="Olympic (Women) - United States" , yes=1, no=0),
                              Opp_Finland = ifelse(x_train$opp_team_name=="Olympic (Women) - Finland" , yes=1, no=0),
                              Opp_Switz = ifelse(x_train$opp_team_name=="Olympic (Women) - Switzerland" , yes=1, no=0),
                              Opp_OAR = ifelse(x_train$opp_team_name=="Olympic (Women) - Olympic Athletes from Russia" , yes=1, no=0),
                              Home = ifelse(x_train$venue=="home" , yes=1, no=0))
x_train_dummies = cbind(x_train_dummies, x_train[,4:20])
y_train_dummies = as.numeric(y_train)-1

x_test_dummies = data.frame(Canada = ifelse(x_test$team_name=="Olympic (Women) - Canada" , yes=1, no=0),
                             USA = ifelse(x_test$team_name=="Olympic (Women) - United States" , yes=1, no=0),
                             Finland = ifelse(x_test$team_name=="Olympic (Women) - Finland" , yes=1, no=0),
                             Switz = ifelse(x_test$team_name=="Olympic (Women) - Switzerland" , yes=1, no=0),
                             OAR = ifelse(x_test$team_name=="Olympic (Women) - Olympic Athletes from Russia" , yes=1, no=0),
                             Opp_Canada = ifelse(x_test$opp_team_name=="Olympic (Women) - Canada" , yes=1, no=0),
                             Opp_USA = ifelse(x_test$opp_team_name=="Olympic (Women) - United States" , yes=1, no=0),
                             Opp_Finland = ifelse(x_test$opp_team_name=="Olympic (Women) - Finland" , yes=1, no=0),
                             Opp_Switz = ifelse(x_test$opp_team_name=="Olympic (Women) - Switzerland" , yes=1, no=0),
                             Opp_OAR = ifelse(x_test$opp_team_name=="Olympic (Women) - Olympic Athletes from Russia" , yes=1, no=0),
                             Home = ifelse(x_test$venue=="home" , yes=1, no=0))
x_test_dummies = cbind(x_test_dummies, x_test[,4:20])
y_test_dummies = as.numeric(y_test)-1

data_train = list(x_train=as.matrix(x_train_dummies),y_train=y_train_dummies)
dtrain <- xgb.DMatrix(data = data_train$x_train, label = data_train$y_train)
data_test = list(x_test=as.matrix(x_test_dummies),y_test=y_test_dummies)
dtest <- xgb.DMatrix(data = data_test$x_test, label = data_test$y_test)

bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 2)
pred <- predict(bst, data_test$x_test)
prediction <- as.numeric(pred > 0.5)
conf_rf5 = mmetric(y_test,prediction,metric="CONF")$conf
conf_rf5
#sensitivity
conf_rf5[1,1]/sum(conf_rf5[1,])
#precision
conf_rf5[1,1]/sum(conf_rf5[,1])
#specificity
conf_rf5[2,2]/sum(conf_rf5[2,])
#accuracy
(conf_rf5[1,1]+conf_rf5[2,2])/sum(conf_rf5)
#test error
mean(prediction != data_test$y_test)


try.iterations <- c(20,25,30,35,40,45,50, 55, 60, 65, 70)
try.eta <- c(0.1, 0.12, 0.15, 0.18, 0.2,0.22,0.25,0.28,0.3,0.32,0.35)
try.bag <- c(0.5, 0.6, 0.65, 0.7, 0.75, 0.8)
xest.mspe <- c()
x.ids <- c()
for(t in 1:length(try.iterations)){
  #for(d in 1:length(try.depth)){
  for(e in 1:length(try.eta)){
    for(b in 1:length(try.bag)){
      x.boost <- xgb.train(data = dtrain,
                         #max_depth=try.depth[d], 
                         eta=try.eta[e],
                         subsample=try.bag[b], 
                         nrounds=try.iterations[t],
                         objective="binary:logistic",
                         verbose = 0,
                         eval_metric='logloss')
      xest.mspe <- c(xest.mspe,mean(as.numeric(predict(x.boost, data_train$x_train) > 0.5) != data_train$y_train))
      x.ids <- c(x.ids, paste(try.iterations[t], try.eta[e], try.bag[b], sep=" "))
    }
  }
  #}
}
chosen <- as.numeric(strsplit(x.ids[which.min(xest.mspe)], " ")[[1]])
x.boost <- xgb.train(data = dtrain,
                     #max_depth=try.depth[d], 
                     eta=chosen[2],
                     subsample=chosen[3], 
                     nrounds=chosen[1],
                     objective="binary:logistic",
                     verbose = 0,
                     eval_metric='logloss')
pred <- predict(x.boost, data_test$x_test)
prediction <- as.numeric(pred > 0.5)
conf_xgb = mmetric(y_test,prediction,metric="CONF")$conf
conf_xgb
#sensitivity
conf_xgb[1,1]/sum(conf_xgb[1,])
#precision
conf_xgb[1,1]/sum(conf_xgb[,1])
#specificity
conf_xgb[2,2]/sum(conf_xgb[2,])
#accuracy
(conf_xgb[1,1]+conf_xgb[2,2])/sum(conf_xgb)
#test error
mean(prediction != data_test$y_test)
