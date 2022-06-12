#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
#%%

data_train = pd.read_csv('train.csv').to_numpy()

x, y = data_train[:, 2:], data_train[:, 1]


#%%
feature = ["linear", "quad", "exp", "cos"]

def transformation(name,x):
    if name == "linear":
        return x
    elif name == "quad":
        return x**2
    elif name == "exp":
        return np.exp(x)
    elif name == "cos":
        return np.cos(x)

#%%
model = Ridge(fit_intercept=False)
x_trans = np.zeros((x.shape[0], 21))
print(x_trans)

#%%
j = 0
for i in feature:
    x_trans[:,j:j+5] = transformation(i,x)
    j = j+5

#%%

x_trans[:,20] = np.ones(x.shape)[:,0]
print(x_trans[3,:])

#%%
ridge = model.fit(x_trans, y)
print(ridge.coef_)

#%%
pd.DataFrame(ridge.coef_).to_csv("result.csv", header=None, index=None)
# %%

# %%
