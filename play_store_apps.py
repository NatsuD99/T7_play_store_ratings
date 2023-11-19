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
# There are multiple apps with same names but the ID is different, so it's fine.
# App Id, will thus be used as our primary key. 
# The 5 records in which the app name are missing, we'll remove those.

# %%
df_clean = df.dropna(subset=['App Name'])
df_clean.isna().sum()
# %% [markdown]
# There are 22883 apps that do not have a rating or a rating count. 
# Our ultimate target is to predict rating of an app. So imputing the missing values, no matter how good, isn't original. 
# With 2+ million data, we can afford to drop 22k records. So, let's do that.
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
# %%
df_clean = df_clean.drop('Installs', axis=1)
# %%
df_clean.isna().sum()
# 32 & 31 instances of NA value in Developer Id and Email respectively
# %%
## Processing the three string variables 'Developer Id','Developer Email' and 'Developer Website'
# 'Developer Website' has 760831 NA values and so I have chosen not to drop those since it might result in significant data loss.
df_clean = df_clean.dropna(subset=['Developer Id','Developer Email'])
df_clean.isna().sum()
# %%[markdown]
# Now let's concentrate on the currency column which has 20 nan values.
# %%
print(df_clean['Currency'].value_counts())
# The various currencies are:
# 'USD': United States Dollar,'CAD': Canadian Dollar
# 'EUR': Euro 'INR': Indian Rupee 'VND': Vietnamese Dong
# 'GBP': British Pound Sterling  'BRL': Brazilian Real
# 'KRW': South Korean Won  'TRY': Turkish Lira
# 'SGD': Singapore Dollar 'AUD': Australian Dollar
# 'ZAR': South African Rand
# We have'XXX':# This is a special code used to denote that no specific currency
# is involved. It's often used in financial contexts to represent a placeholder
# or a non-standard situation.

# We also notice that the vast majority is USD, some XXX and 
# countable number of records for the rest of the currencies. 
# We'll deal with this properly during data cleaning.
# %%[Handling nan values of currency column]
df_clean = df_clean.dropna(subset=['Currency'])
df_clean.isna().sum()

# %%[markdown]
# Preprocessing the size column
print(df_clean['Size'].isna().sum())
df_clean['Size'].unique()
#df_clean['Size']


# %%[markdown]
#Getting the count of apps of various sizes 
countm_M=0
countk_K=0
countg_G=0
count_varieswithdevice_nan =0
for values in df_clean['Size']:
    if 'M' in str(values) or 'm' in str(values):
        countm_M+=1
    elif 'K' in str(values) or 'k' in str(values):
        countk_K+=1
    elif 'Varies with device' in str(values) or str(values)=='nan':
        count_varieswithdevice_nan+=1
    elif 'G' in str(values) or 'g' in str(values):
        countg_G+=1


total_count=countm_M+countk_K+countg_G+count_varieswithdevice_nan
print(total_count)
print(len(df_clean['Size']))

#%%[markdown]
#The various sizes of apps are listed down below
print(f'Apps of size in megabytes are {countm_M}')
print(f'Apps of size in kilobytes are {countk_K}')
print(f'Apps of size in gigabytes are {countg_G}')
print(f'Apps of varibale sizes and also of nan values are {count_varieswithdevice_nan}')




#%%[markdown]
#Here I am coverting apps of sizeM(megabytes) 
# into their corresponding values in kilobytes(k)
def convert_m_to_kb(x):
    if 'M' in x or 'm' in x:
        return pd.to_numeric(x.replace('M', '').replace('m', '').replace(',','')) * 1024
    else:
        return x

# Convert 'M' or 'm' to kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_m_to_kb)
x = (df_clean['Size'] == 'm') | (df_clean['Size'] == 'M')
# Once we have created the boolean mask x, you can use x.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, x.sum() would give me
# the total count of rows where the 'size' column is either 'm' or 'M'
# But since we have converted it should give us 0.
count_of_m_or_M = x.sum()
print(f"Count of 'm' or 'M' in the 'size' column: {count_of_m_or_M}")
df_clean['Size']


#%%[markdown]
# Here I am coverting apps of size k(kilobytes) into 
# numeric value of the given kilobytes(k) in the datframe
def convert_k_to_numeric(x):
    try:
        if 'K' in x or 'k' in x:
            return pd.to_numeric(x.replace('K', '').replace('k', '').replace(',',''))
    except:
        print(x)
    return x
    
# Convert 'K' or 'k' to numeric value of kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_k_to_numeric)
y = (df_clean['Size'] == 'k') | (df_clean['Size'] == 'K')
# Once we have created the boolean mask y, we can use y.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, y.sum() would give me
# the total count of rows where the 'size' column is either 'k' or 'K'
# But since we have converted it should give us 0.
count_of_k_or_K = y.sum()
print(f"Count of 'k' or 'K' in the 'size' column: {count_of_k_or_K}")
df_clean['Size']


# %%[markdown]
# Here I am coverting apps of size G(Gigabytes) 
# into their corresponding values in kilobytes(k)
def convert_g_to_numeric(x):
    if 'G' in x or 'g' in x:
        return pd.to_numeric(x.replace('G', '').replace('g', '').replace(',','')) * (1024**2)
    else:
        return x
# Convert 'G' or 'g' to kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_g_to_numeric)
z = (df_clean['Size'] == 'g') | (df_clean['Size'] == 'G')
# Once we have created the boolean mask z, we can use z.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, z.sum() would give me
# the total count of rows where the 'size' column is either 'g' or 'G'
# But since we have converted it should give us 0.
count_of_g_or_G = z.sum()
print(f"Count of 'g' or 'G' in the 'size' column: {count_of_g_or_G}")
df_clean['Size']

# %%
varieswithdevice_nan=0
for values in df_clean['Size']:
    if str(values) =='Varies with device' or str(values)=='nan':
        varieswithdevice_nan += 1
# After preprocessing , cleaning and converting the above rows 
# to Kilobytes as the base value I have kept the total no of string 
# values for "varies with device" and "nan" untampered    
if count_varieswithdevice_nan == varieswithdevice_nan:
    print("Unaltered before and after preprocessing")

#The Nan values is zero
print(df_clean['Size'].isna().sum())
    



#%%[markdown]

# %%
df_clean['Minimum Android']
df_clean['Minimum Android'].unique()
df_clean['Minimum Android'].value_counts()


# %%
df_clean['Minimum Android'].isna().sum()
# %%
df_clean.isna().sum()
# %%[markdown]
# We notice that Developer website and privacy policy have too much NA values. 
# Released column has 48371 NA values. We are not going to use these columns for model building.
# But, we'll be using these for analysis and feature engineering going ahead.
# So for now, we'll keep the missing values since it doesn't pose an issue.
