import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import random
from danger_model import get_model

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("data/all_powerplays_4-23-22_cleaned_trimmed.csv")
get_model(data=data, weight=1.25, r=500)
# 366 is solid
# res = pd.read_csv('results.csv')

# res[(res.auc_value.notna()) & (res.prop <= 0.3)].sort_values(by = ['test_score', 'auc_value'], ascending=[False, False]).head(15)
# BAD coding practice but I do like my pretty loops

# weight = pd.DataFrame(np.linspace(1, 2.5, num = 16), columns=['wt'])
# weight['j'] = 1
# p = pd.DataFrame(np.linspace(0.25, 1, num=31), columns=['prop'])
# p['j'] = 1

# df = weight.merge(p, how='outer', on = 'j').drop(columns=['j'])
# df['ranking'] = np.arange(stop=len(df))

# np.random.seed(132)

# df['rand'] = random.sample(range(50000), len(df))

# res = pd.DataFrame(columns=['weight', 'prop', 'rand_int', 'auc_value', 'test_score', 'train_score'])

# for z in tqdm(np.arange(0, len(df))):

#     try: 

#         row = df[df.ranking == z]
#         w = float(row.wt.item())
#         pr = float(row.prop.item())
#         r = int(row.rand.item())

#         w, p, r, a, ts, rs = get_model(data=pd.read_csv("data/all_powerplays_4-23-22_cleaned_final.csv"), p=pr, weight=1, r=500)

#         d = {
#             'weight': [w],
#             'prop': [p],
#             'rand_int': [r],
#             'auc_value': [round(float(a), ndigits=4)],
#             'test_score': [round(ts, ndigits=4)],
#             'train_score': [round(rs, ndigits=4)]
#         }

#         res = res.append(pd.DataFrame.from_dict(d))
    
#     except:
#         continue

# res.to_csv('results.csv', index=False)

# res = pd.read_csv('results.csv')

# # res[(res.auc_value.notna()) & (res.prop <= 0.3)].sort_values(by = ['test_score', 'auc_value'], ascending=[False, False]).head(15)

# data=pd.read_csv("data/all_powerplays_4-23-22_cleaned_final.csv")
# get_model(data=data, p = 0.250, weight=1, r= 500)