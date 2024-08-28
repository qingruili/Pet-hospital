import pandas as pd
import pathlib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Clean_Data/hotel_clean.csv")
con = data.copy()
del con['id']
del con['latitude']
del con['longitude']
del con['price']
cormat = con.corr()
round(cormat,2)

plt.figure(figsize=(14, 10), dpi=80)
plot = sns.heatmap(cormat,annot=True)
plt.savefig('images/Correlation.png')