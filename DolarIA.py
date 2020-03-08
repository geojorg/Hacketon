#Description:   This program uses an artificial intelligence to predict dolar prices based on data
#               of the last 80 days. This progam uses (LSTM)
#Install the following libraries for the project
#pip install pandas-datareader
#pip install -U scikit-learn
#pip install -U matplotlib



#Libraries
import math
import pandas_datareader as web
import numpy as num
import pandas
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plot

#Download the COP-USD Data
data = web.DataReader('COP=X',data_source='yahoo',start='2019-03-04', end='2020-03-04')
#Show data stream
print (data)

#Rows and Columns
data.shape

#Plot the closing value history for COP-USD
plot.figure(figsize=(16,8))
plot.title('Close price for USD-COP Last Year')
plot.plot(data['Close'])
plot.xlabel('Date',fontsize=20)
plot.ylabel('Exchange Rate USD-COP',fontsize=20)
#plot.show()

#Dataframe for prediction
dataframe = data.filter(['Close'])
dataset = dataframe.values

#Training for LSTM
training_data_len= math.ceil(len(dataset)*0.8)
print (training_data_len)

#Data Scaling for LSTM
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print (scaled_data)

#Training DataSet
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(20, len(train_data)):
    x_train.append(train_data[i-20:i,0])
    y_train.append(train_data[i,0])
    if i<= 21:
        print (x_train)
        print (y_train)
        print ()
        

#Convertion of the x and y to arrays
x_train, y_train = num.array(x_train), num.array(y_train)

#Data featurec
#// TODO: RESHAPE NOT WORKING TEST OTHER WAY
#x_train = num.reshape(x_train, (val1,val2,1)) 
#print (x_train)
x_train = num.reshape(x_train,(191,20,1))
print (x_train)

model = Sequential()
#//TODO: REFACTOR THIS METHOD
model.add(LSTM(10, return_sequences=True, input_shape=(20,1)))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='main_squared_error'



 