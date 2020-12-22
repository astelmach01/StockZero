import data as data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import metrics
from datetime import date
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

key = 'Z1UK6CGV0MLWS7VQ'
YTD = ((date.today()) - (date(date.today().year, 1, 1)))
# fix output size
daily_Data = data.getDailyData('F')

EMA_fifty = data.getDailyEMA('F', 100)

daily_Data = data.addColumn(daily_Data, 'index')

daily_Data_plot = data.reverse(daily_Data, 'close')

print(daily_Data)

daily_Data_num = daily_Data.to_numpy()

# print(daily_Data_num)
# def getLower(data):
# lower = data.iloc(data['index'], 0)


X = daily_Data['index'].values.reshape(-1, 1)
y = daily_Data['close'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df.plot()

plt.show()

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

prediction = np.array([80]).reshape(-1, 1)
print("Prediction at x=80")
print(y[80])
print(lr.predict(prediction))

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
linear_reg = LinearRegression()
linear_reg.fit(X_poly, y)
plt.scatter(X, y, color="red")
plt.plot(X, linear_reg.predict(poly.fit_transform(X)), color="blue")
plt.show()


ts = TimeSeries(key=key, output_format='pandas')
stonks, meta = ts.get_daily(symbol='F')
stonks = stonks['4. close']
stonks.index = np.arange(0, 100)
stonks.plot()
plt.title("Test")
plt.show()
print(stonks)