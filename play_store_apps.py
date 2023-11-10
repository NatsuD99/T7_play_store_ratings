# %%
# !pip install kaggle
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# %%
# Run this just once, and then, comment out the code in this cell.
od.download_kaggle_dataset("https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps",
                           data_dir="C://Play_Store_Data")


# %%
df = pd.read_csv("C:\Play_Store_Data\google-playstore-apps\Google-Playstore.csv")
df.head()
# %%
df.shape
# %%
