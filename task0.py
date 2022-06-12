#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#%%

data_train = pd.read_csv('train.csv').to_numpy()
df_test = pd.read_csv('test.csv')
data_test = df_test.to_numpy()
#%%
x, y = data_train[:, 2:], data_train[:, 1]

#%%

model = LinearRegression()
model.fit(x, y)

x_test = data_test[:,1:]
id_test = df_test['Id']

y_test = pd.DataFrame(model.predict(x_test))

result = pd.concat([id_test, y_test], axis=1)

#%%
print(id_test)
print(result[:10])
print(pd.DataFrame(result).head(10))
#%%
result.to_csv('result.csv', header=['Id','y'], index = None)


# %%
