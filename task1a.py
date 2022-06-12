#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#%%

data_train = pd.read_csv('train.csv').to_numpy()

x, y = data_train[:, 1:], data_train[:, 0]
print(type(x))
print(y.shape)

#%%
parameter = [1e-1, 1, 10, 100, 200]
j = 0
RMSE = [0,0,0,0,0]



kf = KFold(n_splits=10)
kf.get_n_splits(x)
KFold(n_splits=10, random_state=None, shuffle=False)


#%%
for i in parameter:
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = Ridge(alpha= i).fit(X_train, y_train)
        y_hat = model.predict(X_test)

        RMSE[j] = RMSE[j] + mean_squared_error(y_test, y_hat)**0.5
    RMSE[j] = RMSE[j] / 10
    j = j+1

print(RMSE)

#%%
pd.DataFrame(RMSE).to_csv("result.csv", header=None, index=None)



# %%
