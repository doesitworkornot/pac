import numpy as np
import pandas as pd

print('Task 1')
df = pd.DataFrame(np.random.rand(10, 5))
print(df)
print(df[df > 0.3].mean(axis=1))


print('\nTask 2')
df2 = pd.read_csv('data/lab5/wells_info.csv', index_col=0)  # Reading from csv

# Converting to datetime format
df2['CompletionDate'] = pd.to_datetime(df2['CompletionDate'])
df2['FirstProductionDate'] = pd.to_datetime(df2['FirstProductionDate'])

# Making group by basin name
df_ = df2.groupby(by=['BasinName']).agg({'CompletionDate': ['max'], 'FirstProductionDate': ['min']})

# Counting difference in month between start and end of each well
df2['diff'] = (df2['CompletionDate'] - df2['FirstProductionDate']).dt.days//30
print(df2)

# Counting difference in month between start and end of each bassin
df_['diff'] = (df_[('CompletionDate', 'max')] - df_[('FirstProductionDate', 'min')]).dt.days//30
print(df_)


print('\nTask 3')
df3 = pd.read_csv('data/lab5/wells_info_na.csv', index_col=0)  # Reading from csv

# Head on variant
# df3['LatWGS84'] = df3['LatWGS84'].fillna(df3['LatWGS84'].mean())
# df3['LonWGS84'] = df3['LonWGS84'].fillna(df3['LonWGS84'].mean())
# df3['CompletionDate'] = df3['CompletionDate'].fillna(df3['CompletionDate'].mode().iloc[0])
# df3['BasinName'] = df3['BasinName'].fillna(df3['BasinName'].mode().iloc[0])

a = [column for column in df3.columns if df3[column].dtypes == 'O']
b = [column for column in df3.columns if df3[column].dtypes != 'O']
# for i in a:
#     print(i)
#     df3[i] = df3[i].fillna(df3[i].mode().iloc[0])
# for i in b:
#     df3[i] = df3[i].fillna(df3[i].mean())

df3[a] = df3[a].fillna(df3[a].mode().iloc[0])
df3[b] = df3[b].fillna(df3[b].mean())
print(df3.head())

