# %% [markdown]
## Import Libraries
# %%
# !pip install kaggle
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# %% [markdown]
## Import Data
# %%
# Run this just once, and then, comment out the code in this cell.
# od.download("https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps")


# %%
df = pd.read_csv("google-playstore-apps/Google-Playstore.csv")
df.head()
# df.tail()
# %% [markdown]
# Data Explorartion
# %%
df.shape
# %%
df.columns
# %%
df.describe()
# %%
df.describe(include='O')
# %% [markdown]
# We notice the count of app name and unique names doesn't match, so either there are duplicated values or missing values.
# But we see app ID having all unique values. Let's see.
# %%
df.isna().sum()
# %% [markdown]
# So, there are 5 app names missing but all the app IDs are present.
# %%
df[df['App Name'].isna()]
# %%
df['App Name'].duplicated().sum()
# %%
# %%
df['App Id'].isna().sum()
# %%
df['App Id'].duplicated().sum()
# %%
df[df['App Name'].duplicated()][['App Name', 'App Id']].head(10)

# %% [markdown]
# So, we notice that there are no missing app IDs and none of the app IDs are duplicated as well.
# There are multiple apps with same names but the ID is different, so it fine.
# App Id, will thus be used as our primary key.

#%%
#Treating missing values for privacy col
sample_values = df['Privacy Policy'].sample(n=20) #overview of sample 
print(sample_values)
df['Privacy Policy'].fillna('Nan', inplace=True)

# %%
#Treating missing values for Relesed col
#replaced missing values with median date
sample_values1 = df['Released'].sample(n=20)
print(sample_values1)
df['Released'] = pd.to_datetime(df['Released'], errors='coerce')
median_date = df['Released'].median()  # Calculate the median date
df['Released'].fillna(median_date, inplace=True)

#%%
