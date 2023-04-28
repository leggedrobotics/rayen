import glob, os
import pandas as pd

os.chdir("./results")
all_pkl=[]
for file in glob.glob("*.pkl"):
    all_pkl.append(file)


all_pkl = sorted(all_pkl)
# print(all_pkl)

all_dataframes=[]
for file in all_pkl:
    all_dataframes.append(pd.read_pickle(file))

df = pd.concat(all_dataframes)

df.to_csv('./merged.csv')  
