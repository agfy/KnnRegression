from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn import model_selection
import numpy as np

boston = datasets.load_boston()
boston.data = scale(boston.data)
arr = np.linspace(1, 10, 200)
k_fold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

p_max = 0.0
summ_max = -10000000.0
for p in arr:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    score_arr = model_selection.cross_val_score(neigh, boston.data, boston.target, cv=k_fold, scoring='neg_mean_squared_error')
    summ = 0.0
    for val in score_arr:
        summ += val

    summ /= len(score_arr)
    if summ > summ_max:
        summ_max = summ
        p_max = p