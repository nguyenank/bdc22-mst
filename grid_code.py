from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
import bisect

# code that breaks the hockey rink up into 80 different boxes

df = pd.read_csv('all_powerplays_4-7-22.csv')
df['x_coord'] = df.x_coord - 100
df['y_coord'] = df.y_coord - 42.5

xbreaks = np.linspace(start=-100, stop=100, num=11)
ybreaks = np.linspace(-42.5, 42.5, num=8)

arr = np.zeros((len(xbreaks), len(ybreaks)), dtype=int)
# arr[0, :] = np.arange(len(ybreaks))
arr[:, 0] = np.arange(len(xbreaks))
arr[:, 1] = np.arange(len(xbreaks)) + 11
arr[:, 2] = np.arange(len(xbreaks)) + 22
arr[:, 3] = np.arange(len(xbreaks)) + 33
arr[:, 4] = np.arange(len(xbreaks)) + 44
arr[:, 5] = np.arange(len(xbreaks)) + 55
arr[:, 6] = np.arange(len(xbreaks)) + 66
arr[:, 7] = np.arange(len(xbreaks)) + 77

bisect.bisect_right(df.x_coord, arr.all())


class Grid():
    def __init__(self,x,y,grid_len):
        self.total_x = x
        self.total_y = y
        self.grid_len = grid_len
        self.node_list=[]
        for y in range(self.total_y//self.grid_len):
             for x in range(self.total_x//self.grid_len):
                self.node_list.append((x*self.grid_len,y*self.grid_len))

    def return_node_at_given_x_y(self,x,y):
        node_index = y*(self.total_y//self.grid_len) + x
        print(self.node_list[node_index])


my_grid = Grid(100,100,5)
print(my_grid.return_node_at_given_x_y(8,6))

Grid(-100, 100, 11)



self.total_x = x
self.total_y = y
self.grid_len = grid_len
self.node_list=[]
for y in range(self.total_y//self.grid_len):
        for x in range(self.total_x//self.grid_len):
        self.node_list.append((x*self.grid_len,y*self.grid_len))

x = -100
y = 100
grid_len = 11

node_list = []

for y in range(y//grid_len):
    for x in range(x//grid_len):
        node_list.append((x * grid_len, y * grid_len))

pd.DataFrame(np.linspace(start=-100, stop=100, num=11), columns=['x']).merge(pd.DataFrame(np.linspace(start=-100, stop=100, num=11), columns=['x']), how='outer')

def find_intersection(values, intervals):

    intervals

    # https://stackoverflow.com/questions/33114624/looking-for-corresponding-intervals-for-points
    output = []
    value_index = 0
    interval_index = 0

    while value_index < len(values) and interval_index < len(intervals):
        current_value = values[value_index]
        current_interval = intervals[interval_index]
        lower_bound, upper_bound = current_interval

        if current_value < lower_bound:
            output.append((None, current_value))
            # This value cannot belong to any greater interval.
            value_index += 1
        elif current_value > upper_bound:
            # No other value can belong to this interval either.
            interval_index += 1
        else:
            output.append((current_interval, current_value))
            # At most one value per interval and one interval per value.
            value_index += 1
            interval_index += 1

    # If we ran out of intervals all remaining values do not belong to any.
    for v in values[value_index:]:
        output.append((None, v))

    return output

find_intersection(values=['50', '75'], intervals=np.linspace(start=-100, stop=100, num=11))