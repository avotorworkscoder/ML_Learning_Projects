#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

#%%
# Simple Data Analysis of Employer Dataset
df = pd.DataFrame({
    "age": [22, 25, 28, 32, 36, 40, 45, 50],
    "experience": [0, 2, 4, 6, 10, 14, 18, 25],
    "salary": [25000, 30000, 38000, 48000, 70000, 90000, 120000, 150000],
    "department": ["HR", "HR", "Sales", "Sales", "Sales", "AI", "AI", "AI"]
})

# This tells pandas to show all columns instead of truncating them
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df.info()
df.describe(include='all')

sns.scatterplot(x="experience", y="salary", hue="department", data=df)
plt.show()

# Salary, Experience & Department Analysis
# 1. Download the latest version
path = kagglehub.dataset_download("gmudit/employer-data")
print("Path to dataset files:", path)

# 2. List files to find the exact CSV name
files = os.listdir(path)
print("Files in folder:", files)

# 3. Load the CSV into a DataFrame (df)
# Assuming the file is named 'employer_data.csv' or similar
csv_path = os.path.join(path, files[0]) 
df = pd.read_csv(csv_path)

# Inspect the DataFrame
print(df.head())
print(df.info())
print(df.describe(include='all'))

# 4. Plot using Seaborn
df["nor_salary"] = (df["Salary"] - df["Salary"].min()) / (df["Salary"].max() - df["Salary"].min())
sns.scatterplot(x="Experience_Years", y="Salary", data=df, hue="nor_salary", style="Gender")

plt.title("Experience vs Salary by Department")
plt.show()
#%%
markers = {"Male": ".", "Female": "X"}
sns.scatterplot(x="Experience_Years", y="Salary", data=df, hue="Age", size="nor_salary", sizes=(5, 50), style="Gender", markers=markers)

plt.title("Experience vs Salary by Department")
plt.show()
#%%
sns.relplot(
    x="Experience_Years", y="Salary", data=df, 
    col="Education_Level", hue="Department", style="Department",
    kind="scatter")
plt.show()





#%%
#Customer Purchase Behavior & Outliers
df = pd.DataFrame({
    "customer_id": range(1, 11),
    "purchase_amount": [120, 150, 180, 200, 220, 250, 300, 350, 5000, 5200],
    "visits_per_month": [1, 2, 2, 3, 3, 4, 4, 5, 2, 2]
})
sns.histplot(df["purchase_amount"], kde=True)
plt.show()

sns.boxplot(y=df["purchase_amount"])
plt.show()

sns.scatterplot(x="visits_per_month", y="purchase_amount", data=df)
plt.show()

#%%
# EDA on Amazon Sales Dataset
dt = pd.read_csv("F:\Python code/amazon.csv")

print(dt.head())
print(dt.info())
print(dt.describe(include='all'))   

sns.histplot(data=dt, x="Review Rating", bins=40, kde=True)
plt.show()

sns.histplot(data=dt, x="rating", log_scale=True, kde=True)
plt.show()





# Time Series Analysis of Sensor Data
#%%
time = pd.date_range("2024-01-01", periods=60, freq='min')

df = pd.DataFrame({
    "timestamp": time,
    "temperature": np.random.normal(30, 1.5, 60),
    "vibration": np.random.normal(0.02, 0.004, 60)
}).set_index("timestamp")

print( df.head() )
df.plot(subplots=True, figsize=(10, 5))
plt.show()

df["temp_roll"] = df["temperature"].rolling(5).mean()
df[["temperature", "temp_roll"]].plot()
plt.show()

sns.histplot(df["vibration"], kde=True)
plt.show()

# %%



