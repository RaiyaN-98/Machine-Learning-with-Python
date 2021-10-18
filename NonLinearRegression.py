import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("china_gdp.csv")
# print(df.head(10))

# plt.figure(1, figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
# plt.plot(x_data, y_data, 'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')

## Building the model

def sigmoid(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1*(x - beta_2)))
    return y

# Checking a curve fit
beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1, beta_2)

plt.figure(1)
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# normalize our data
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)

# using curve_fit to fit curve to our data using least square method
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(2, figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
 
plt.show()

## Finfing out the accuracy of our model

# split the data 
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat, test_y))
