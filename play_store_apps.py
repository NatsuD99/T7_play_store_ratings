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
# od.download_kaggle_dataset("https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps",
#                            data_dir="C://Play_Store_Data")


# %%
df = pd.read_csv("C:\Play_Store_Data\google-playstore-apps\Google-Playstore.csv")
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
df['App Name'].duplicated()
# %%
