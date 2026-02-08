import pandas as pd

df = pd.DataFrame({
    'user_id': [1, 2, 2, 3, 4, 5, 5],
    'age':[25, None, None, 40, 120, 30, 30],
    'salary':[50000, 60000, 60000, None, 70000, 0, 0],
    'city':['delhi', 'mumbai', 'mumbai', None, 'kolkata', 'chennai', 'chennai'],
    'purchase_amount':[250, 300, 300, 400000, None, 150, -150],
})
print("Original DataFrame:")
print(df)
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Description:")
print(df.describe())


def clean_pipeline(df):
    df = df.copy()
    # Handle missing values
    df['age']=df['age'].fillna(df['age'].median())
    df['salary']=df['salary'].fillna(df['salary'].median())
    df['salary']=df['salary'].replace(0, df['salary'].median())
    df['city']=df['city'].fillna(df['city'].mode()[0])
    df['purchase_amount']=df['purchase_amount'].fillna(0)
    
    # handle duplicate entries
    df=df.drop_duplicates(subset=['user_id'], keep='last')
    # Handle outliers in age
    df.loc[df['age'] > 100, 'age'] = df['age'].median()
    # Handle outliers in purchase_amount
    df.loc[(df['purchase_amount'] < 0), 'purchase_amount'] = 0
    df.loc[(df['purchase_amount'] > df['salary']), 'purchase_amount'] = df['salary']
    
    # datatype fixes
    df['age'] = df['age'].astype(int)
    df['salary'] = df['salary'].astype(int)
    df['city'] = df['city'].astype('category')
    
    return df


# Detect missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Handle missing values
df['age']=df['age'].fillna(df['age'].median())
print("\nDataFrame after handling missing age values:")
print(df['age'])

df['salary']=df['salary'].fillna(df['salary'].median())
df['salary']=df['salary'].replace(0, df['salary'].median())
print("\nDataFrame after handling salary values:")
print(df['salary'])

df['city']=df['city'].fillna(df['city'].mode()[0])
print("\nDataFrame after handling city values:")        
print(df['city'])

df['purchase_amount']=df['purchase_amount'].fillna(0)
print("\nDataFrame after handling purchase_amount values:") 
print(df['purchase_amount'])

# handle duplicate entries
df=df.drop_duplicates(subset=['user_id'], keep='last')
print("\nDataFrame after removing duplicates:")
print(df)

# Detect outliers using IQR method
Q1 = df[['age', 'salary', 'purchase_amount']].quantile(0.25)
Q3 = df[['age', 'salary', 'purchase_amount']].quantile(0.75)
IQR = Q3 - Q1
print("\nIQR values:")
print(IQR)
outliers = ((df[['age', 'salary', 'purchase_amount']] < (Q1 - 1.5 * IQR)) | (df[['age', 'salary', 'purchase_amount']] > (Q3 + 1.5 * IQR)))
print("\nOutliers detected using IQR method:")
print(outliers)

# Handle outliers in age
df.loc[df['age'] > 100, 'age'] = df['age'].median()
print("\nDataFrame after handling outliers in age:")
print(df['age'])

# Handle outliers in purchase_amount
df.loc[(df['purchase_amount'] < 0), 'purchase_amount'] = 0
df.loc[(df['purchase_amount'] > df['salary']), 'purchase_amount'] = df['salary'].median()
print("\nDataFrame after handling outliers in purchase_amount:")
print(df['purchase_amount'])

# datatype fixes
print("\nData types before conversion:")
print(df.dtypes)
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(int)   
df['city'] = df['city'].astype('category')
print("\nData types after conversion:")
print(df.dtypes) 

print("\nCleaned DataFrame:")
print(df)

log = {}

log["missing_age"] = df["age"].isnull().sum()
log["duplicates"] = df.duplicated().sum()
print("\nData Cleaning Log:")
print(log)