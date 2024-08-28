import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# this script produces graphs of the correlation between the number of parks to the price and the ratings.
hotel = pd.read_csv("Clean_Data/hotel_clean.csv")
#print(hotel)

hotel = hotel[hotel['price'] < 1000]

# ----------------- NUM PARKS vs PRICE ---------------
reg2 = stats.linregress(hotel['num_parks'],hotel['price'])
print('P-value of relationship between price and number of parks: ',reg2.pvalue)
hotel['prediction_price'] = hotel['num_parks']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['num_parks'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['num_parks'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['num_parks'],hotel['price'], alpha=0.1)
plt.plot(hotel['num_parks'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Num Parks')
plt.ylabel('Price')
plt.savefig('images/price_and_NumParks.png')


# ----------------- NUM MARKETS vs PRICE ---------------
reg2 = stats.linregress(hotel['num_market'],hotel['price'])
print('P-value of relationship between price and number of markets: ',reg2.pvalue)
hotel['prediction_price'] = hotel['num_market']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['num_market'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['num_market'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['num_market'],hotel['price'], alpha=0.1)
plt.plot(hotel['num_market'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Num Market')
plt.ylabel('Price')
plt.savefig('images/price_and_NumMarket.png')

# ----------------- NUM BusStops vs PRICE ---------------
reg2 = stats.linregress(hotel['num_bustop'],hotel['price'])
print('P-value of relationship between price and number of BusStops: ',reg2.pvalue)
hotel['prediction_price'] = hotel['num_bustop']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['num_bustop'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['num_bustop'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['num_bustop'],hotel['price'], alpha=0.1)
plt.plot(hotel['num_bustop'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Num BusStops')
plt.ylabel('Price')
plt.savefig('images/price_and_NumBusStops.png')

# ----------------- Min Dist PARKS vs PRICE ---------------
reg2 = stats.linregress(hotel['dis_parks'],hotel['price'])
print('P-value of relationship between price and Min Dist PARKS: ',reg2.pvalue)
hotel['prediction_price'] = hotel['dis_parks']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['dis_parks'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['dis_parks'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['dis_parks'],hotel['price'], alpha=0.1)
plt.plot(hotel['dis_parks'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Min Dist to a Park')
plt.ylabel('Price')
plt.savefig('images/price_and_MinDistToPark.png')


# ----------------- Min Dist  MARKETS vs PRICE ---------------
reg2 = stats.linregress(hotel['dis_market'],hotel['price'])
print('P-value of relationship between price and Min Dist  MARKETS: ',reg2.pvalue)
hotel['prediction_price'] = hotel['dis_market']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['dis_market'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['dis_market'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['dis_market'],hotel['price'], alpha=0.1)
plt.plot(hotel['dis_market'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Min Dist to Market')
plt.ylabel('Price')
plt.savefig('images/price_and_MinDistToMarket.png')

# ----------------- Min Dist  BusStops vs PRICE ---------------
reg2 = stats.linregress(hotel['dis_bustop'],hotel['price'])
print('P-value of relationship between price and Min Dist  BusStops: ',reg2.pvalue)
hotel['prediction_price'] = hotel['dis_bustop']*reg2.slope + reg2.intercept

data2_new = pd.DataFrame({'y': hotel['price'], 'x': hotel['dis_bustop'], 'one': 1})
results2 = sm.OLS(data2_new['y'], data2_new[['x', 'one']]).fit()
print(results2.summary())

hotel['residuals'] = hotel['price'] - (reg2.slope*hotel['dis_bustop'] +reg2.intercept ) 
print(reg2.rvalue**2)
print(reg2.rvalue)
print(reg2.slope)

plt.figure()
plt.scatter(hotel['dis_bustop'],hotel['price'], alpha=0.1)
plt.plot(hotel['dis_bustop'], hotel['prediction_price'], 'r-', linewidth=3)
plt.xlabel('Min dist to a BusStop')
plt.ylabel('Price')
plt.savefig('images/price_and_MinDistBusStop.png')