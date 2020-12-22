from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# if short term MA crosses below long term it is bearish (SELL)
# if shorter term MA crosses above long term it is bullish (BUY)


key = 'Z1UK6CGV0MLWS7VQ'
ts = TimeSeries(key=key, output_format='pandas')
ti = TechIndicators(key=key, output_format='pandas')

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


def getDailyData(q):
    x, y = ts.get_daily(symbol=q)
    return formatDailyData(x)


def formatDailyData(data):
    new_data = data.drop(['5. volume', '1. open', '2. high', '3. low'], axis=1)
    new_data = new_data[:: -1]

    numbers = np.arange(100)
    numbers = numbers[:: -1]
    # rename all row values to how many days from start

    # 0 is most recent day
    # 99 is 99 days ago

    new_data.index = numbers

    # 4. close becomes close
    new_data.rename(columns=lambda x: x[3:], inplace=True)

    return new_data


def getDailyEMA(symbol, period):
    x, y = ti.get_ema(symbol=symbol, interval='daily', time_period=period, series_type='close')
    x = formatEMA_Data(x)
    return x


def formatEMA_Data(ema_data):
    ema_data = ema_data.tail(100)
    numbers = np.arange(100) 
    numbers = numbers[:: -1]
    # rename all row values to how many days from start

    # 0 is most recent day
    ema_data.index = numbers
    return ema_data


def reverse(thingy, column_name):
    thingy[column_name] = thingy[column_name].values[::-1]
    return thingy


def addColumn(data, new_column):
    numbers = []
    for i in range(100):
        numbers.append(i)
    numbers = numbers[:: -1]

    data[new_column] = numbers

    return data

