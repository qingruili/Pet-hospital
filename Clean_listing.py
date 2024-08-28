import pandas as pd
import pathlib
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler



def distance(data1, stations):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [data1['longitude'], data1['latitude'], stations['lon'], stations['lat']])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def num_st(data1, stations,n):
    stations['distance'] = distance(data1, stations)
    num = len(stations[stations['distance']<=n])
    return num


def dis_st(data1, stations):
    stations['distance'] = distance(data1, stations)
    dis = stations['distance'].min().round(2)
    return dis


input_directory = pathlib.Path('Origin_Data')
output_directory = pathlib.Path('CLean_Data')


data = pd.read_csv(input_directory /'listings.csv')[['id','name','latitude', 'longitude', 'price','minimum_nights','room_type','reviews_per_month']]
data_skytrain = pd.read_csv(output_directory /'Skytrain_clean.csv')
data_BusStop = pd.read_csv(output_directory /'BusStop_clean.csv')
data_Restaurant = pd.read_csv(output_directory /'Restaurant_clean.csv')
data_Park = pd.read_csv(output_directory /'Parks_clean.csv')
data_Supermarket = pd.read_csv(output_directory /'Supermarket_clean.csv')

data = data[data['room_type']== 'Entire home/apt']


match1= data['name'].str.extract(r'(\S*) in Vancouver · \S+(\d.\d+|New) · (\d bedroom|Studio)',expand=False)


del data['name']
del data['room_type']
data = pd.concat([data, match1[[1,2]]], axis=1)
data = data.rename(columns={1: 'rate', 2: 'num_bedroom'})
match2= data['num_bedroom'].str.extract(r'(\d) bedroom',expand=False)
data['num_bedroom'] = match2
data['num_bedroom'] = np.where(data['num_bedroom'].isnull(), 1,data['num_bedroom'])

print(data.price.describe([0.25,0.5,0.75,0.99]))

data = data[data['price'] <=1170.88]
data = data[data['minimum_nights'] <= 365]
data['num_skytrain'] = data.apply(num_st, stations=data_skytrain,n=1000,axis=1)
data['num_bustop'] = data.apply(num_st, stations=data_BusStop,n=500,axis=1)
data['num_restaurant'] = data.apply(num_st, stations=data_Restaurant,n=500,axis=1)
data['num_parks'] = data.apply(num_st, stations=data_Park,n=500,axis=1)
data['num_market'] = data.apply(num_st, stations=data_Supermarket,n=500,axis=1)

data['dis_skytrain'] = data.apply(dis_st, stations=data_skytrain,axis=1)
data['dis_bustop'] = data.apply(dis_st, stations=data_BusStop,axis=1)
data['dis_restaurant'] = data.apply(dis_st, stations=data_Restaurant,axis=1)
data['dis_parks'] = data.apply(dis_st, stations=data_Park,axis=1)
data['dis_market'] = data.apply(dis_st, stations=data_Supermarket,axis=1)


data['reviews_per_month'] = np.where(data['reviews_per_month'].isnull(), 0,data['reviews_per_month'])
data_train = data[data['rate'] != 'New']
data_train = data_train[data_train['rate'].notna()]
data_va = data[(data['rate'] == 'New') | (data['rate'].isnull())]


model = make_pipeline(
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=9)
)

data_train['rate'] = pd.to_numeric(data_train['rate'], downcast="float")

model.fit(data_train[['price','minimum_nights','num_bedroom',
                      'num_skytrain','num_bustop','num_restaurant','num_parks','num_market',
                      'dis_skytrain','dis_bustop','dis_restaurant','dis_parks', 'dis_market']],data_train['rate'])

predict = model.predict(data_va[['price','minimum_nights','num_bedroom',
                      'num_skytrain','num_bustop','num_restaurant','num_parks','num_market',
                      'dis_skytrain','dis_bustop','dis_restaurant','dis_parks', 'dis_market']])

data_va['rate'] = predict
data_new = pd.concat([data_train, data_va])



save_path = os.path.join(output_directory, f"hotel_clean.csv")
data_new.to_csv(save_path, index=False)