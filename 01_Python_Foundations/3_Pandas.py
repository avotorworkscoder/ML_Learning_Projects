# %%
import pandas as pd

#%%
df=pd.read_csv('Electronics Price chart.csv',encoding='utf-8',sep=',')
print(df.head())

#%%
print(df.info())

#%%
print(df.describe())

#%%
print(df.tail())

#%%
print(df.columns)

#%%
print(df.dtypes)

#%%
print(df.shape)

#%%
print(df.size)

#%%
print(type(df))

# %%
print(df['Name'].head())

# %%
print(df[['Name', 'Market Price']].head())
# %%
print(df['M. Total'].fillna(0)> 500)
# %%
print(df['Name']=="Motor wheels")
# %%
print(df[df['M. Total'].fillna(0).astype(int) > 1000])

print(df.sort_values('M. Total', ascending=False).head(3))


print(df.tail())
df.drop(93, inplace=True)
print(df.tail())
print(df.groupby('Quantity')['M. Total'].sum())

df.info()