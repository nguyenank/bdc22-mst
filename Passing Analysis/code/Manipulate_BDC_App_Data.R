# Load (install if needed) the required packages
load.libraries = c("tidyverse","readr","ggplot2","RColorBrewer","knitr","tictoc", "here","dplyr","data.table", "ggExtra", "ggtext", "patchwork", "paletteer", "scales","randomForest","jcolors")
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for (libs in install.lib) {install.packages(libs, dependencies = TRUE)}
sapply(load.libraries, require, character = TRUE)

all_pps_reduced <- read_csv("H:/Hockey/Current Git/Big-Data-Cup-2022-Private/all_powerplays_new_method_11-24-22.csv")

i=1
row = all_pps_reduced[i,]
puck_x = row$x_coord %>% unlist()
x = row %>% select(home_x_ft_1,home_x_ft_2,home_x_ft_3,home_x_ft_4,home_x_ft_5,home_x_ft_6,
                   away_x_ft_1,away_x_ft_2,away_x_ft_3,away_x_ft_4,away_x_ft_5,away_x_ft_6) %>% unlist()
vx = row %>% select(home_vel_x_1,home_vel_x_2,home_vel_x_3,home_vel_x_4,home_vel_x_5,home_vel_x_6,
                    away_vel_x_1,away_vel_x_2,away_vel_x_3,away_vel_x_4,away_vel_x_5,away_vel_x_6) %>% unlist()
puck_y = row$y_coord %>% unlist()
y = row %>% select(home_y_ft_1,home_y_ft_2,home_y_ft_3,home_y_ft_4,home_y_ft_5,home_y_ft_6,
                   away_y_ft_1,away_y_ft_2,away_y_ft_3,away_y_ft_4,away_y_ft_5,away_y_ft_6) %>% unlist()
vy = row %>% select(home_vel_y_1,home_vel_y_2,home_vel_y_3,home_vel_y_4,home_vel_y_5,home_vel_y_6,
                    away_vel_y_1,away_vel_y_2,away_vel_y_3,away_vel_y_4,away_vel_y_5,away_vel_y_6) %>% unlist()

position = row %>% select(home_goalie_1,home_goalie_2,home_goalie_3,home_goalie_4,home_goalie_5,home_goalie_6,
                          away_goalie_1,away_goalie_2,away_goalie_3,away_goalie_4,away_goalie_5,away_goalie_6)
goalie = which(position=='Goalie')
puck = which.min((puck_x-x)^2+(puck_y-y)^2)
if(substr(names(puck),1,4)=='home'){
  off = c(rep(1,6),rep(-1,6))
  goalie = which(unlist(position[7:12]))
}else{
  off = c(rep(-1,6),rep(1,6))
  goalie = which(unlist(position[1:6]))
}
if(length(goalie)==0){
  goalie = NA
} 
if(length(puck)==0){
  puck = NA
} 
row = as.matrix(c(unlist(row),x,y,vx,vy,off,puck,goalie),nrow=1)
all_pps_reduced_new = t(row)
colnames(all_pps_reduced_new) = c(colnames(all_pps_reduced),names(x),names(y),names(vx),names(vy),
                                  'off_1','off_2','off_3','off_4','off_5','off_6','off_7',
                                  'off_8','off_9','off_10','off_11','off_12','puck','goalie')

for(i in 2:nrow(all_pps_reduced)){
  print(i)
  row = all_pps_reduced[i,]
  puck_x = row$x_coord %>% unlist()
  x = row %>% select(home_x_ft_1,home_x_ft_2,home_x_ft_3,home_x_ft_4,home_x_ft_5,home_x_ft_6,
                     away_x_ft_1,away_x_ft_2,away_x_ft_3,away_x_ft_4,away_x_ft_5,away_x_ft_6) %>% unlist()
  vx = row %>% select(home_vel_x_1,home_vel_x_2,home_vel_x_3,home_vel_x_4,home_vel_x_5,home_vel_x_6,
                      away_vel_x_1,away_vel_x_2,away_vel_x_3,away_vel_x_4,away_vel_x_5,away_vel_x_6) %>% unlist()
  puck_y = row$y_coord %>% unlist()
  y = row %>% select(home_y_ft_1,home_y_ft_2,home_y_ft_3,home_y_ft_4,home_y_ft_5,home_y_ft_6,
                     away_y_ft_1,away_y_ft_2,away_y_ft_3,away_y_ft_4,away_y_ft_5,away_y_ft_6) %>% unlist()
  vy = row %>% select(home_vel_y_1,home_vel_y_2,home_vel_y_3,home_vel_y_4,home_vel_y_5,home_vel_y_6,
                      away_vel_y_1,away_vel_y_2,away_vel_y_3,away_vel_y_4,away_vel_y_5,away_vel_y_6) %>% unlist()
  
  position = row %>% select(home_goalie_1,home_goalie_2,home_goalie_3,home_goalie_4,home_goalie_5,home_goalie_6,
                            away_goalie_1,away_goalie_2,away_goalie_3,away_goalie_4,away_goalie_5,away_goalie_6)
  goalie = which(position=='Goalie')
  puck = which.min((puck_x-x)^2+(puck_y-y)^2)
  if(substr(names(puck),1,4)=='home'){
    off = c(rep(1,6),rep(-1,6))
    goalie = which(unlist(position[7:12]))
  }else{
    off = c(rep(-1,6),rep(1,6))
    goalie = which(unlist(position[1:6]))
  }
  if(length(goalie)==0){
    goalie = NA
  } 
  if(length(puck)==0){
    puck = NA
  } 
  row = as.matrix(c(unlist(row),x,y,vx,vy,off,puck,goalie),nrow=1)
  all_pps_reduced_new = rbind(all_pps_reduced_new,t(row))
}
to_save = as.data.frame(all_pps_reduced_new)
write_csv(to_save,file='all_powerplays_clean.csv')

#### To Python
with_metrics <- read_csv("H:/Hockey/Current Git/Big-Data-Cup-2022-Private/data_w_metrics.csv")
all_data = cbind(to_save,with_metrics)
write_csv(all_data,file='all_data.csv')
