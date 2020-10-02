import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize']=(20.0,10.0)

data=pd.read_csv('headbrain.csv')
# print(data.shape)
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

m = len(X)
X=X.reshape((m,1))


reg=LinearRegression()

reg=reg.fit(X,Y)

Y_pred = reg.predict(X)
mse = mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print(rmse)
r2_score= reg.score(X,Y)

print(r2_score)