import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    chi2
)

def variable_selection(x_w_inter, y, new_names, k = 40):
    new_names = np.array(new_names)

    selection = SelectKBest(chi2, k = k)
    trans_x = selection.fit_transform(x_w_inter, y)
    raw_selected_names = new_names[selection.get_support()]

    raw_names = []

    for i in raw_selected_names:
        raw_names.append(i)

    return trans_x, raw_names