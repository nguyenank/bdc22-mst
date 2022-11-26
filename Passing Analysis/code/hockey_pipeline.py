import numpy as np
import pandas as pd
from scipy.stats import norm
from numba import jit
from numpy.typing import ArrayLike

MAX_TIME = 2
EPS = 1e-7
GG = 32.174

METRICS = ['prob','rink_ctrl','best_case','expected']
TIME_PENALTY = 0.1
MAX_VEL=35.5   # maximum skater velocity in ft/sec
ALPHA = 1.3 # acceleration coefficient (not directly acceleration, but more like a speed decay)
TR = 0.189 # reaction time (based on the article Phil sent) 
MM = 0.1 # Coefficient of friction between puck and ice, I'll find the source for this
BETA_PUCK = 0.1322 # Puck air drag coefficient (actuall it's the coefficient divided by the mass so beta = k/m if k is the drag coefficient)
BETA_CTRL = 2.5 # pitch control coefficient used as beta in ice_ctrl_xyt and teamwise_ice_ctrl_xyt, taken from the Spearman paper
X_DECAY = 2000 #value used as decay_x
Y_DECAY = 500 #value used as decay_y
GOALIE_DIST = 8 # maximum reasonable distance for goalie to go away from goal
GLX = 11 # Goalie X coord
GLY = 42.5  # Goalie Y coord
STICK = 5 # Stick length 
TARGET_RADIUS = 27.5

@jit(nopython  = True)
def inside_boards(x: np.ndarray,y: np.ndarray, t:np.ndarray,
                  target_radius: float = TARGET_RADIUS):
    radius = (x<28) * ((y>57)*((x-28)**2 + (y-57)**2)**0.5 + (y<28)*((x-28)**2 + (28-y)**2)**0.5)
    ix = (radius<=target_radius) * (0<x) * (x<100) * (0<y) * (y<85)
    return x[ix],y[ix],t[ix]

class tracks():
    def __init__(self,
                x: ArrayLike, # x locations of players (array or list of floats) 
                y: ArrayLike, # y locations of players (array or list of floats)
                vx: ArrayLike, # x velocity of players (array or list of floats)
                vy: ArrayLike, # y velocity of players (array or list of floats)
                goalie: int, # column number for goalie
                puck: int, # column number for which player has the puck
                off: ArrayLike, # array or list of integers +1 for offence, -1 for defence (or true and false)
                vp: float = 55,
                phi_res: float = 0.01,
                t_res: float = 0.01,
                # metric: str = 'expected'
                ):
        assert len(set([len(x),len(y),len(vx),len(vy),len(off)]))<=2
        # if not metric in METRICS:
            # raise ValueError('Metric choice is not in recognized metric list, please choose another metric')
        # self.metric = metric
        self.xp = x[puck]
        self.yp = y[puck]
        self.puck = puck
        self.phi_res = phi_res
        self.off = np.where(np.array(off)==1,1,-1)
        self.t_res = t_res
        self.vp = vp
        self.x = np.array(x)
        self.y = np.array(y)
        self.vx = np.array(vx)
        self.vy = np.array(vy)
        self.goalie = goalie
        # self.tracks = pd.DataFrame({'x':x,'y':y,'vx':vx,'vy':vy,'goalie':goalie,'off':off})
        self.player_motion()
        self.grid = np.concatenate([self.one_pass(self, phi)() for phi in np.arange(-np.pi,np.pi+EPS, phi_res)], axis = 0)

    def player_motion(self, alpha: float = ALPHA, t_r: float = TR, vmax: float = MAX_VEL):
        t = np.arange(self.t_res,MAX_TIME, self.t_res).reshape(-1,1)
        # x = self.x.reshape(-1,1)
        # vx = self.vx.reshape(-1,1)
        # y = self.y.reshape(-1,1)
        # vy = self.vy.reshape(-1,1)

        self.c_x = np.where(t<t_r, self.x + self.vx * t, self.x + t_r * self.vx + self.vx * (1-np.exp(-alpha * (t-t_r))/alpha))
        self.c_y = np.where(t<t_r, self.y + self.vy * t, self.y + t_r * self.vy + self.vy * (1-np.exp(-alpha * (t-t_r))/alpha))
        self.r = np.where(t<t_r,0,vmax * (t -t_r - (1-np.exp(-alpha * (t-t_r)))/alpha)) 


    class one_pass():
        def __init__(self, outer_self: 'tracks', phi: float):
            self.t_res = outer_self.t_res
            self.phi = phi
            self.x0 = outer_self.xp
            self.y0 = outer_self.yp
            self.vp = outer_self.vp
            self.x,self.y,self.t = self.make_grid()
            self.outside_creese = (self.x-GLX)**2 + (self.y-GLY)**2 > GOALIE_DIST**2
            self.get_metric(outer_self)#, outer_self.metric)
        
        def make_grid(self):
            t = np.arange(self.t_res,MAX_TIME,self.t_res)
            x, y = self.puck_motion_model(t)
            return inside_boards(x,y,t)
            

        def puck_motion_model(self,t: np.ndarray,
                                    mu:float = MM, 
                                    beta: float = BETA_PUCK, 
                                    g: float = GG):
                vx = self.vp*np.sin(self.phi)
                vy = -self.vp*np.cos(self.phi)
                
                x =  self.x0 + (vx + mu*g * vx/self.vp/beta) * (1 - np.exp(-beta * t))/beta - (mu*g * t * vx/self.vp)/beta
                y = self.y0 + (vy + mu*g * vy/self.vp/beta) * (1 - np.exp(-beta * t))/beta - (mu*g * t * vy/self.vp)/beta
                
                return x, y
        
        def score_prob(self, decay_x = X_DECAY, decay_y = Y_DECAY):
            # Scoring Probability function 
            x = self.x
            y = self.y
            self.score = (np.abs((x-11)/((42.5-y)**2+(11-x)**2)**0.5)+1)/np.where(x<11,8,4)*np.exp(-((11-x)**2/decay_x +(42.5-y)**2/decay_y))


        def dist_to_xyt(self,outer_self: 'tracks'):
                # If time is smaller than reaction time, skater keeps going at initial speed
            ln = len(self.t)
            tx = self.x.reshape(-1,1)
            ty = self.y.reshape(-1,1)

            remaining_dist = ((tx-outer_self.c_x[:ln,:])**2 + (ty-outer_self.c_y[:ln,:])**2)**0.5-outer_self.r[:ln,:]
            return(np.maximum(remaining_dist,EPS))
        
                
        def get_metric(self,outer_self: 'tracks'):  #, metric: str = 'prob'):
            # dists = np.array([self.dist_to_xyt(x0,y0,vx,vy) for x0,y0,vx,vy in zip(outer_self.x, outer_self.y, outer_self.vx, outer_self.vy)]).T
            dists = self.dist_to_xyt(outer_self)
            dists[self.outside_creese,outer_self.goalie]  = np.maximum(dists[self.outside_creese,outer_self.goalie], ((self.x[self.outside_creese]-GLX)**2 + (self.y[self.outside_creese]-GLY)**2)**0.5 - GOALIE_DIST)
            ctrl =(dists/MAX_VEL)**(-BETA_CTRL)* outer_self.off.reshape(1,-1)
            self.all_ctrl = ctrl.sum(1)/np.abs(ctrl).sum(1)
            # if metric == 'rink_ctrl':
                # self.metric = self.all_ctrl
                # return 0
            
            dists = np.delete(dists, outer_self.puck,axis = 1)
            off_mat = np.delete(outer_self.off, outer_self.puck)
            base_probs = self.t_res * (norm.cdf(dists/STICK+1)-norm.cdf(dists/STICK-1))/TIME_PENALTY * (1 - np.exp(-self.t.reshape(-1,1)/(TR + TIME_PENALTY * off_mat.reshape(1,-1))))
            ranks = (-base_probs).argsort()
            # print(ranks.shape, base_probs.shape)
            ranked_probs = np.take_along_axis(base_probs,ranks,1)
            off_mat = off_mat[ranks]
            # print(off_mat)
            # print(np.concatenate((np.ones((1,dists.shape[1])),1-ranked_probs[:-1,:]),0).cumprod(1).shape)
            adj_probs = np.concatenate((np.ones((1,dists.shape[1])),1-ranked_probs[:-1,:]),0).cumprod(1)*ranked_probs
            # print(adj_probs.shape)
            adj_pass_off = (adj_probs*(off_mat==1)).sum(1)
            # pass_def = adj_probs[~off_mat] # Not sure we actually need this line
            missed = 1 - adj_probs.sum(1)
            missed = np.append(1,missed[:-1]).cumprod()
            pass_off = adj_pass_off * missed
            # if metric == 'prob':
            #     self.metric = pass_off.sum()  * np.ones(self.x.shape)
            # if metric == 'best_case':
            #     self.score_prob()
            #     adj_pass_value = self.score*all_ctrl*adj_pass_off
            #     self.metric = adj_pass_value.max() * np.ones(self.x.shape)

            # elif metric == 'expected':
            #     self.score_prob()
            #     loc_pass_value = self.score*all_ctrl*pass_off
            #     self.metric = loc_pass_value.sum() * np.ones(self.x.shape)
            self.prob = pass_off.sum()  * np.ones(self.x.shape)
            self.score_prob()
            adj_pass_value = self.score*self.all_ctrl*adj_pass_off
            self.best_case = adj_pass_value.max() * np.ones(self.x.shape)
            loc_pass_value = self.score*self.all_ctrl*pass_off
            self.expected = loc_pass_value.sum() * np.ones(self.x.shape)

        def __call__(self):   
            return np.stack((self.x,self.y,self.t,self.all_ctrl,self.score*self.all_ctrl,self.prob,self.best_case,self.expected),1)  #Robyn - added self.score*self.all_ctrl for location value of passer  

#Robyn - added metrics which uses tracks to calculate various metrics used in modelling
class metrics(tracks):
    def home_plate(self):
        y_upper = 35.05+0.95*self.grid[:,0]
        y_lower = 49.95-0.95*self.grid[:,0]
        square = np.array((self.grid[:,0]<=46,self.grid[:,0]>=11,
                        self.grid[:,1]<=64.5,self.grid[:,1]>=20.5,
                        np.logical_and(self.grid[:,0]<=31,self.grid[:,1]<=y_upper),
                        np.logical_and(self.grid[:,0]<=31,self.grid[:,1]>=y_lower)))
        in_square = np.logical_and.reduce(square)
        square_grid = self.grid[in_square,]
        return np.mean(square_grid[:,3])

    def control_of_rink(self):
        return np.mean(self.grid[:,3])

    def max_metrics(self,n):
        return np.max(self.grid[:,n])

    def passer_location(self):
        x_passer = self.xp+self.vx[self.puck]
        y_passer = self.yp+self.vy[self.puck]
        return self.grid[np.argmin((x_passer-self.grid[:,0])**2+(y_passer-self.grid[:,1])**2),4]

    def metrics_offense(self,measure='max'):
        x_0 = np.delete(np.array(self.x)+self.vx,self.puck)
        y_0 = np.delete(np.array(self.y)+self.vy,self.puck)
        off_0 = np.delete(self.off,self.puck)
        x_off = x_0[np.array(off_0)==1]
        y_off = y_0[np.array(off_0)==1]
        vals_at_players = np.empty((len(x_off),3))
        i=0
        for (xx,yy) in zip(x_off,y_off):
            vals_at_players[i,:]=(self.grid[np.argmin((xx-self.grid[:,0])**2+(yy-self.grid[:,1])**2),5:])
            i+=1
        #row 0 = x, row 1 = y, col 1 = successful, col 2 = best, col 3 = expected
        positions = pd.DataFrame([x_off[np.argmax(vals_at_players,axis=0)],y_off[np.argmax(vals_at_players,axis=0)]])
        if measure=='max':
            return (tuple(vals_at_players.max(axis=0)),tuple(positions[0]),tuple(positions[1]),tuple(positions[2]))
        else:
            return tuple(vals_at_players.mean(axis=0))
        
    def get_metrics(self):   
        metrics_grid = (self.home_plate(),self.control_of_rink(),
        self.max_metrics(5),self.max_metrics(6),self.max_metrics(7),
        self.passer_location(), #is the passer in a shooting position with control
        self.metrics_offense('max'), #is there an available pass for each of these options
        self.metrics_offense('mean'))   #how good is the teams positioning
        return np.array(metrics_grid[0:6]+metrics_grid[6][0][:]+metrics_grid[6][1][:]+metrics_grid[6][2][:]+metrics_grid[6][3][:]+metrics_grid[7][:])


        

if __name__ == '__main__':
    x = list(200 -np.array([171.4262, 155.6585, 153.7146, 150.5869, 156.3463, 179.8383, 180.8131, 186.6146, 179.9982]))
    y= list(np.array([49.31514, 48.25991, 70.17542, 13.65429, 28.51970, 38.44596, 36.80571, 38.32781, 22.03946]))
    vx=list(np.array([6.725073,  4.964445, -3.097599, 14.252625,  4.286796,  1.925091, -2.295729, -0.294258,  6.464229]))
    vy=list(np.array([-7.1037417,  -7.9677960,  -6.4446342,   6.5618985, -10.9455216,  -4.7444208,  -4.1465373,  -0.3377985, -5.4265284]))
    goalie = 7
    puck= 3
    off=list(np.array([-1, -1, 1, 1, -1, 1, -1, -1, 1]))
    all_tracks = metrics(x,y,vx,vy,goalie,puck,off) #Robyn - changed tracks to metrics

#print(all_tracks.get_metrics())

    #Robyn - all_tracks.grid to get x,y,t,control,location value, successful, best case, expected
    #Robyn - all_tracks.get_metrics() will give you home_plate_control percent, rink control percent, max successful, max best case, max expected, 
    #value at passer, 
    #max successful at player locations and player location's x and y, max best case at player locations and player location's x and y, 
    #max expected at player locations and player location's x and y, mean successful, mean best case, mean expected
    

    # Robyn - Code for getting metrics on training data

#     csv_df = pd.read_csv('all_powerplays_clean.csv')
#     metrics_grid = np.empty((csv_df.shape[0],18))
#     current_index = -1
#     for data in csv_df.iterrows():
#     current_index+=1
#     # if current_index==24:
#     #     data['away_x_5_velo']=0.05
#         # data[1]['away_y_5_velo']=0.05
#     x0 = data[1][125:137]
#     to_remove = np.where(pd.isna(x0),1,0)
#     x = list((np.array(x0))[np.array(to_remove)==0])
#     y = list((np.array(data[1][137:149]))[np.array(to_remove)==0])
#     vx_0 = np.where(pd.isna(data[1][149:161]),0.05,data[1][149:161])
#     vx = list((np.array(vx_0))[np.array(to_remove)==0])
#     vy_0 = np.where(pd.isna(data[1][161:173]),0.05,data[1][161:173])
#     vy = list((np.array(vy_0))[np.array(to_remove)==0])
#     off = list((np.array(data[1][173:185]))[np.array(to_remove)==0])
#     puck = int(data[1][185])-1
#     puck= puck-sum(to_remove[:puck])
#     if pd.isna(data[1][186]): #goalie not in tracking data
#         x = x+[12]
#         y = y+[42.5]
#         vx = vx+[0.01]
#         vy = vy+[0.01]
#         off = off+[-1]
#         goalie = len(x)-1
#     else:
#         goalie = int(data[1][186])-1
#         goalie= goalie-sum(to_remove[:goalie])
#     # Puck is outside the offensive zone or not enough players tracked
#     if x[puck]>105 or y[puck]>90 or off.count(-1)<2 or off.count(1)<2 or current_index in (391,392):
#         metrics_grid[current_index,:] = np.repeat(np.nan,18)
#     else:
#         # Small adjustment if puck is just outside, to account for tracking uncertainty
#         if x[puck]>99:
#         x[puck]=99
#     if x[puck]<1:
#       x[puck]+=1
#     if y[puck]<1:
#       y[puck]+=1
#     if y[puck]>84:
#       y[puck]=84
#     all_tracks = metrics(x,y,vx,vy,goalie,puck,off)
#     metrics_grid[current_index,:] = all_tracks.get_metrics()
#   print(current_index)
