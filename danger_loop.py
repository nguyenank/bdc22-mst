import pandas as pd
import numpy as np
from danger_model import get_model

weight = np.linspace(1, 2.5, num = 16)
p = np.linspace(0.25, 1, num=31)

w, p, r, a, ts, rs = get_model(data=pd.read_csv("data/all_powerplays_4-23-22_cleaned_final.csv"), p=pr, weight=1, r=500)