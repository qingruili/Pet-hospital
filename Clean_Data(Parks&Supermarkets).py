import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import pyspark as spark
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

def getDataFromCenter(x):
    if not pd.isna(x.center):
        x['lat'] = x.center['lat']
        x['lon'] = x.center['lon']
    return x

def main():
    conversion = pd.read_json(sys.argv[1], orient='records')
    print(conversion)
    conversion = pd.DataFrame.from_records(conversion['data'])
    print(conversion.columns)
    conversion = conversion.dropna(subset=['tags'])
    print(conversion)
    conversion['name'] = pd.DataFrame.from_records(conversion['tags'])['name']
    if 'center' in conversion.columns:
        conversion = conversion.apply(getDataFromCenter,axis=1)
        conversion = conversion.drop(columns=['nodes', 'type', 'members', 'center', 'tags'])
        print(conversion)
    else:
        conversion = conversion.drop(columns=['nodes', 'type', 'tags'])
        print(conversion)
    print(conversion)
    conversion = conversion.dropna(subset=['lat'])
    print(conversion)
    print(conversion.iloc[0])
    #conversion = conversion.dropna()
    #print(conversion['data'].apply(lambda x: pd.DataFrame.from_dict(x)[['lat','lon']]))
    #pd.Series(conversion).to_csv(sys.argv[2], index=False, header=False)
    
    conversion.to_csv(sys.argv[2], index=False)
    

if __name__ == '__main__':
    main()
