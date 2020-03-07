#Description:   This program uses an artificial intelligence to predict dolar prices based on data
#               of the last 80 days. This progam uses (LSTM)
#Install the following libraries for the project
#pip install pandas-datareader
#pip install -U scikit-learn
#pip install -U matplotlib



#Libraries
import math
import pandas_datareader as web
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plot

#Download the COP-USD Data
data = web.DataReader('COP=X',data_source='yahoo',start='2018-12-19', end='2020-01-06')
#Show data stream
print (data)

#Rows and Columns
data.shape

#Plot the closing value history for COP-USD
plot.figure(figsize=(16,8))
plot.title('Close price for USD-COP Last 2 Years')
plot.plot(data['Close'])
plot.xlabel('Date',fontsize=20)
plot.ylabel('Exchange Rate USD-COP',fontsize=20)
plot.show()

 