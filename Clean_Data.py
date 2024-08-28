import re
import os
import pandas as pd


def extract(data):
    clean = pd.DataFrame(columns=['id','lon','lat'])
    clean['id'] = data['features'].apply(lambda x: x['id'])
    clean['id'] = clean['id'].apply(lambda x: re.sub(r'\D', '', x))
    temp = data['features'].apply(lambda x: x['geometry']['coordinates'])
    for index, value in temp.items():
        if len(value) <= 1:
            temp[index] = value[0][0]
    clean['lon'] = temp.apply(lambda x: x[0])   
    clean['lat'] = temp.apply(lambda x: x[1])
    return clean


def __main__():
    save_dir = "Clean_Data"

    file1 = r'Origin_Data/BusStop.geojson'
    file2 = r'Origin_Data/Skytrain.geojson'
    file3 = r'Origin_Data/Restaurant.geojson'

    files = [file1,file2,file3]
    
    for file in files:
        file_name = re.search(r'\/([^\/]+)\.', file).group(1)
        clean = extract(pd.read_json(file))
        print("File: " + file + " has been cleaned")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{file_name}_clean.csv")
        clean.to_csv(save_path, index=False)

if __name__ == '__main__':
    __main__()