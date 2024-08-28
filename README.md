# CMPT353-Hotel-Grader
## Used Libraries
- NumPy
- Pandas
- Matplotlib
- Scala
- Seborne
- SciPy
- OS
- RE
- SKLearn
- StatsModel
## Original Data
All raw data is stored in the [Origin_Data](https://github.com/IAsianRice/CMPT353-Hotel-Grader/tree/master/Origin_Data) Folder.
- BusStop.geojson
- Parking.json
- Restautant.geojson
- Skytrain.geojson
- Supermarkets.json
- listing.csv
## Cleaning Data
All the cleaning data is stored in the [Clean_Data](https://github.com/IAsianRice/CMPT353-Hotel-Grader/tree/master/Origin_Data) Folder.

We use the file [Clean_Data.py](https://github.com/IAsianRice/CMPT353-Hotel-Grader/blob/master/Clean_Data.py) to clean the bus stop, restaurants, and skytrains:
- BusStop_clean.csv
- Restautant_clean.csv
- Skytrain_clean.csv

We use the file [Clean_Data(Parks&SUpermarkets).py](https://github.com/IAsianRice/CMPT353-Hotel-Grader/blob/master/Clean_Data(Parks%26Supermarkets).py) to clean the parks and Supermarkets:
- Supermarkets_clean.csv
- Parks_clen.csv

We use the file [Clean_listing.py](https://github.com/IAsianRice/CMPT353-Hotel-Grader/blob/master/Clean_listing.py) to clean the Airbnb hotel data and calculate the extra data we need:
- hotel_clean.csv

## Analyze Data
We use the files below to analyze the relationship between variables and hotel prices.
- Analysis_Parks_POI.py
- Analysis_Restaurant_nights_skytrain.ipynb
- Analysis_room.ipynb
- Analysis_rate_reviews.py

## Correlation
We use [cor.py](https://github.com/IAsianRice/CMPT353-Hotel-Grader/blob/master/cor.py) to find the correlation between varibles.
## Machine Learning
We use the [prediction.py](https://github.com/IAsianRice/CMPT353-Hotel-Grader/blob/master/prediction.py) to find the most suitable model to estimate the hotel prices.

## image
All the images we get from code are stored in the [images](https://github.com/IAsianRice/CMPT353-Hotel-Grader/tree/master/images) folder.

## Running Code
Run the code in this order:
- Cleaning Data Python files
  - Do hotel_cleaning.py last
- Analysis Data Python files
- For correlation, run cor.py
- For ml model and prediction, run prediction.py

For all the .py files, you can just use (while in directory, or targeting this directory):
```
python3 file_name.py
```
