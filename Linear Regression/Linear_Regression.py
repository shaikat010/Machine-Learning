
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('Area_Price_Dataset.csv')


df.describe()

print(df)

#area is the independent var
# price is the variable var

plt.xlabel('Area')
plt.ylabel('Price(US$)')
plt.scatter(df.area,df.prices, color = 'red', marker = '+')

#plt.show()

#the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.prices)

#prediction
print("This is the predicted price for 500 Area")
print( reg.predict([[500]]))


#The model coefficient and intercept
print(reg.coef_)
print(reg.intercept_)


d = pd.read_csv('areas.csv')
print(d)

#this will return me the prices for the given areas
prices = reg.predict(d)
print(prices)

d['prices'] = prices

#we can see here the prices are shown
print(d)

# puts the d dataframe in the other csv file
d.to_csv('Area_Price_Prediction_Output.csv')




#need to check this code
# plt.scatter(df.area,df.prices, color = 'red', marker = "+")
# plt.plot(df.area, reg.predict([[df.area]]),color = 'blue')
# plt.show()