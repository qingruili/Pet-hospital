import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression




OUTPUT_TEMPLATE = (
    'Name:                      Train  Valid\n'    
    'Linear Regression:         {model_li_tr:.3f}  {model_li_va:.3f}\n'
    'KNN Regression:            {model_knn_tr:.3f}  {model_knn_va:.3f}\n'
    'Random forest Regression:  {model_ran_tr:.3f}  {model_ran_va:.3f}\n'
    'Neural Network Regression: {model_nnet_tr:.3f}  {model_nnet_va:.3f}'
)

data = pd.read_csv("Clean_Data/hotel_clean.csv")

X_train, X_valid, y_train, y_valid = train_test_split(data[['rate','minimum_nights','num_bedroom',
                      #'num_skytrain','num_bustop','num_restaurant','num_parks','num_market',
                      'reviews_per_month','dis_skytrain','dis_bustop','dis_restaurant','dis_parks', 'dis_market'
                      ]],data['price'])



## linear Regression

model_li = LinearRegression(fit_intercept=False)
one = np.ones((len(y_train),1))
X_with = np.concatenate([one,X_train],axis=1)
model_li.fit(X_with,y_train)

ones = np.ones((len(y_valid),1))
X_with_va = np.concatenate([ones,X_valid],axis=1)




# Knn
model_knn = make_pipeline(
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=40)
)
model_knn.fit(X_train,y_train)



# Random Forest
model_ran =RandomForestRegressor(500, max_depth=6,min_samples_leaf=15)
model_ran.fit(X_train, y_train)


# netural nnets
model_nnet = MLPRegressor(random_state=1,max_iter=600)
model_nnet.fit(X_train, y_train)


print(OUTPUT_TEMPLATE.format(
        model_li_tr=model_li.score(X_with, y_train),
        model_li_va=model_li.score(X_with_va, y_valid),
        model_knn_tr=model_knn.score(X_train, y_train),
        model_knn_va=model_knn.score(X_valid, y_valid),
        model_ran_tr=model_ran.score(X_train, y_train),
        model_ran_va=model_ran.score(X_valid, y_valid),
        model_nnet_tr=model_nnet.score(X_train, y_train),
        model_nnet_va=model_nnet.score(X_valid, y_valid),
    ))

#https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/

Z = ['LinearRegression','KNN','RandomForest','Natural Network']
Ygirls = [model_li.score(X_with, y_train),
          model_knn.score(X_train, y_train),
          model_ran.score(X_train, y_train),
          model_nnet.score(X_train, y_train)]
Zboys = [model_li.score(X_with_va, y_valid),
         model_knn.score(X_valid, y_valid),
         model_ran.score(X_valid, y_valid),
         model_nnet.score(X_valid, y_valid)]
  
X_axis = np.arange(len(Z))


plt.figure(figsize=(10, 10), dpi=80)
plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Train')
plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Valid')
  
plt.xticks(X_axis, Z)
plt.xlabel("Models")
plt.ylabel("Scores(R^2)")
plt.title("Score compared between models")
plt.legend()
#plt.show()
plt.savefig('images/Score.png')