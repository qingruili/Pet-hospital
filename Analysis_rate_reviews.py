import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



data = pd.read_csv("Clean_Data/hotel_clean.csv")
data['num_bedroom'] = pd.to_numeric(data['num_bedroom'], downcast="integer")
data['rate'] = pd.to_numeric(data['rate'], downcast="float")
data['reviews_per_month'] = pd.to_numeric(data['reviews_per_month'], downcast="float")


# price and rate analysis
num = data.copy()
reg = stats.linregress(num['rate'],num['price'])
print('P-value of relationship between price and rate: ',reg.pvalue)
num['prediction_price'] = num['rate']*reg.slope + reg.intercept
num['residuals'] = num['price'] - (reg.slope*num['rate'] +reg.intercept )
print(reg.rvalue**2)

plt.figure()
plt.plot(num['rate'],num['price'], "b.", alpha=0.2)
plt.plot(num['rate'], num['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Rate')
plt.ylabel('Price')
plt.savefig('images/price_and_rate.png')


# price and number of reviews in each hotel analysis
data3 = data.copy()
reg3 = stats.linregress(data3['reviews_per_month'],data3['price'])
print('P-value of relationship between price and reviews per month: ',reg3.pvalue)
data3['prediction_price'] = data3['reviews_per_month']*reg3.slope + reg3.intercept
data3['residuals'] = data3['price'] - (reg3.slope*data3['reviews_per_month'] +reg3.intercept ) 
print(reg3.rvalue**2)

plt.figure()
plt.plot(data3['reviews_per_month'],data3['price'], "b.", alpha=0.5)
plt.plot(data3['reviews_per_month'], data3['prediction_price'], 'r-', linewidth=3)
plt.xlabel('reviews_per_month')
plt.ylabel('Price')
plt.savefig('images/price_and_reviews_per_month.png')

