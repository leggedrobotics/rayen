import glob, os
import pandas as pd

os.chdir("./results")
all_pkl=[]
for file in glob.glob("dataset*.pkl"):
    all_pkl.append(file)
all_pkl = sorted(all_pkl)
all_dataframes=[]
for file in all_pkl:
    all_dataframes.append(pd.read_pickle(file))


df = pd.concat(all_dataframes)


df=df.drop_duplicates(subset=['method'])
# print(df)


results2d = df[df['method'].str.startswith('dataset2d')]
results3d = df[df['method'].str.startswith('dataset3d')]

results2d=results2d.set_index('method')
results3d=results3d.set_index('method')


results2d["[In dist] loss"]/=(results2d.loc["dataset2d_Optimization"].at["[In dist] loss"]) #Normalize the cost
results2d["[Out dist] loss"]/=(results2d.loc["dataset2d_Optimization"].at["[Out dist] loss"]) #Normalize the cost

results3d["[In dist] loss"]/=(results3d.loc["dataset3d_Optimization"].at["[In dist] loss"]) #Normalize the cost
results3d["[Out dist] loss"]/=(results3d.loc["dataset3d_Optimization"].at["[Out dist] loss"]) #Normalize the cost



df = pd.concat([results2d, results3d])

df=df.rename(columns={"[In dist] loss": "[In dist] n.loss", "[Out dist] loss": "[Out dist] n.loss"})


print(df)

# results2d['[In dist] loss']=results2d['[In dist] loss']/results2d[ results2d['method']=='dataset2d_Optimization'  ]

df.to_csv('./merged.csv')  
