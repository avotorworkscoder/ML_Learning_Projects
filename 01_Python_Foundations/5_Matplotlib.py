#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    "age": [25, 35, 42, 29, 38, 45, 31],
    "salary": [50000, 45000, 90000, 50000, 65000, 120000, 60000],
    "department": ["AI", "HR", "AI", "Sales", "Sales", "AI", "HR"],
    "experience": [2, 8, 15, 4, 10, 20, 6]
})

plt.plot(df["experience"], df["salary"])
plt.xlabel("Experience (years)")
plt.ylabel("Salary ($)")
plt.title("Experience vs Salary")
plt.grid(True)
plt.show()

sns.scatterplot(
    x="experience",
    y="salary",
    hue="department",
    data=df
)
plt.title("Experience vs Salary by Department")
plt.show()


plt.hist(df["salary"], bins=5, edgecolor='black')
plt.xlabel("Salary ($)")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()

sns.histplot(df["salary"], kde=True)
plt.title("Salary Distribution with KDE")
plt.show()


sns.boxplot(
    x="department",
    y="salary",
    data=df
)
plt.title("Salary Distribution by Department")
plt.show()


corr = df[["age", "salary", "experience"]].corr()
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.show()

# 1. Distribution
sns.histplot(df["salary"], kde=True)

# 2. Outliers
sns.boxplot(x="department", y="salary", data=df)

# 3. Relationship
sns.scatterplot(x="experience", y="salary", hue="department", data=df)

# 4. Correlation
sns.heatmap(df[["age", "salary", "experience"]].corr(), annot=True)
plt.show()

#%%
sns.relplot(
    x="experience",
    y="salary",
    hue="department",
    size="age",
    sizes=(20, 200),
    data=df,
    kind="scatter"
)
plt.show()

# %%
sns.relplot(
    x="experience",
    y="salary",
    col="department",
    data=df,
    kind="scatter"
)
plt.show()

# %%
sns.lmplot(
    x="experience",
    y="salary",
    data=df
)
plt.show()

# %%
sns.kdeplot(
    data=df,
    x="salary",
    hue="department",
    fill=True
)
plt.show()

# %%
sns.violinplot(
    x="department",
    y="salary",
    data=df
)
plt.show()

# %%
sns.pairplot(
    df[["age", "salary", "experience"]],
    diag_kind="kde"
)
plt.show()
# %%

#%%
#7 Visualization for Feature Engineering Validation
sns.histplot(df["salary"], label="Raw", color="red")
sns.histplot(df["salary_norm"], label="Normalized", color="blue")
plt.legend()
plt.title("Salary Distribution Before and After Normalization")
plt.show()

# %%
# Before & After Cleaning
df_before = pd.DataFrame({
    "salary": [50000, 52000, 55000, 60000, 65000, 70000, 300000],
    "experience": [2, 3, 4, 5, 6, 7, 5],
    "department": ["AI", "AI", "HR", "HR", "Sales", "Sales", "AI"]
})
df_after = df_before.copy()
df_after["salary"] = df_after["salary"].clip(
    upper=df_after["salary"].quantile(0.95)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(y=df_before["salary"], ax=axes[0])
axes[0].set_title("Before Cleaning")

sns.boxplot(y=df_after["salary"], ax=axes[1])
axes[1].set_title("After Cleaning")

plt.show()

# %%
# Feature Engineering Validation
df_after["salary_norm"] = (
    df_after["salary"] - df_after["salary"].min()
) / (
    df_after["salary"].max() - df_after["salary"].min()
)
sns.histplot(df_after["salary"], label="Raw", color="red", kde=True)
sns.histplot(df_after["salary_norm"], label="Normalized", color="blue", kde=True)
plt.legend()
plt.title("Raw vs Normalized Salary Distribution")
plt.show()

# %%
#Leakage Detection (Time Series)
dates = pd.date_range("2024-01-01", periods=20)
df_ts = pd.DataFrame({
    "date": dates,
    "sales": np.random.randint(100, 300, size=20)
}).set_index("date")

train_end = "2024-01-14"

plt.figure(figsize=(10,4))
plt.plot(df_ts.index, df_ts["sales"], label="Sales")
plt.axvline(pd.to_datetime(train_end), color="red", linestyle="--", label="Train/Test Split")
plt.legend()
plt.title("Time Series Leakage Check")
plt.show()

# %%
# Model Error & Residual Analysis
y_true = np.array([50, 55, 60, 65, 70])
y_pred = np.array([52, 54, 58, 68, 75])
residuals = y_true - y_pred

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red")
plt.xlabel("Predicted Value")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Analysis")
plt.show()

# %%
# Class Imbalance
df_cls = pd.DataFrame({
    "churn": [0,0,0,0,0,0,1]
})

sns.countplot(x="churn", data=df_cls)
plt.title("Class Distribution")
plt.show()

# %%
