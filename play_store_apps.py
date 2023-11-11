# %% [markdown]
## Import Libraries
# %%
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# %% [markdown]
## Import Data
# %%
# Uncomment, run this just once, and then, comment out the code in this cell.
# od.download("https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps")


# %%
df = pd.read_csv("google-playstore-apps/Google-Playstore.csv")
df.head()

# %% [markdown]
## Data Explorartion
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
## Handling Missing values 
# So, we notice that there are no missing app IDs and none of the app IDs are duplicated as well.
# There are multiple apps with same names but the ID is different, so it fine.
# App Id, will thus be used as our primary key. 
# The 5 records in which the app name are missing, we'll remove those.

# %%
df_clean = df.dropna(subset=['App Name'])
df_clean.isna().sum()
# %% [markdown]
# There are 22883 apps that do not have a rating or a rating count. 
# Our ultimate target is to predict rating of an app. So imputing the missing values, no matter how good, isn't original. 
# With 2+ million data, we can affort to drop 22k records. So, let's do that.
# %%
df_clean = df_clean.dropna(subset=['Rating'])
df_clean = df_clean.dropna(subset=['Rating Count'])
# %%
print(df_clean[['Installs', 'Minimum Installs']].dtypes)
df_clean[['Installs', 'Minimum Installs']].head()
# %% [markdown]
# The values in minimum installs are object type with '+' appended at the end. 
# It also has commas.
# It also seems like the values in the 2 columns are same.
# Let's remove those and have `Installs` as a numerical column and then compare 
# to check if the values are actually the same.
# %%
df_clean['Installs']=df_clean['Installs'].map(lambda x: x[:-1])
df_clean['Installs']= df_clean['Installs'].map(lambda x: x.replace(',', ''))
df_clean['Installs']= pd.to_numeric(df_clean['Installs'])
# %%
df_clean['Installs'].equals(df_clean['Minimum Installs'].astype('int64'))
# %% [markdown]
# So both the columns have same values. The meaning of both the feature is also more or less the same.
# So, we will drop one of them.
df_clean = df_clean.drop('Installs', axis=1)
# %%
df_clean.isna().sum()
# %%
