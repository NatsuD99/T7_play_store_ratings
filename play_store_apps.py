# %% [markdown]
## Import Libraries
# %%
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tabulate import tabulate
import plotly.express as px
sns.set_palette('ocean_r')

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
# %%[markdown]
#Since we have 6526 rows consisting of nan we can afford
# to drop it .Considering the amount of data we have it should not posess a problem
df_clean = df_clean.dropna(subset=['Minimum Android'])
df_clean.isna().sum()


df_clean.isna().sum()
# %%[markdown]
# We notice that Developer website and privacy policy have too much NA values. 
# Released column has 48371 NA values. We are not going to use these columns for model building.
# But, we'll be using these for analysis and feature engineering going ahead.
# So for now, we'll keep the missing values since it doesn't pose an issue.

# %%[markdown]
#Exploratory Data Analysis
#1-Data Cleaning
#2-Feature Engineering
#3-Visualisation
#[These are the three things will focus on in Eda]
             
# %%
print(df_clean['Category'].nunique())
sns.histplot(data=df_clean, x='Category', kde=True, )
plt.xticks(rotation = 90, size = 6)
plt.show()
# We notice that there are 48 different categories, and education category has the highest count.
# The distribution is very uneven with a right tail.
# There are many categories with very less values.
# Let's take a look at the values once
# %%
df_clean['Category'].value_counts(normalize=True)
# On having a clearer look at the data we notice that there are some categories with minor
# spelling changes which are the same. Ex: 'Education' and 'Educational'.
# Lets clean such categories by combining the values into one.
# %%
df_clean['Category'] = df_clean['Category'].str.replace('Educational', 'Education')
df_clean['Category'] = df_clean['Category'].str.replace('Music & Audio', 'Music')
# %%
# Now let's take a look at the top 10 categories
top10cat = ['Education', 'Music', 'Business', 'Tools', 
            'Entertainment', 'Lifestyle', 'Books & Reference',
            'Personalization', 'Health & Fitness', 'Productivity']
df_top10cat= df_clean[df_clean['Category'].isin(top10cat)]
df_top10cat['Category'].value_counts(normalize=True).plot.barh()
plt.show()
# %% [markdown]
# So the Education category apps take up about 20% of all the apps.
# %%
# Now let's take a look at the app ratings.
sns.histplot(x='Rating', data=df_clean, bins=20, kde=True)
plt.show()
# Interestingly, we observe that a huge number of apps have 0 rating.
# Let's 1st get a better visual by omitting those
# %%
sns.histplot(x='Rating', data=df_clean[df_clean['Rating']>0], 
             bins=20, kde=True)
plt.show()
# We get a better idea from this that most of the apps have a rating b/w 3.5-5

# %%
# Now let's check the number of ratings.
df_clean['Rating Count'].describe().apply('{:.5f}'.format)
# Whoa!! Too high standard deviation and 75% of data is below 42 
# while the maximum vlaue is greater than 1.3 million.
# Let's 1st try to see the entire plot and then we'll see only the ones
# below 42 to get a better idea
# %%
sns.histplot(x='Rating Count', data=df_clean, bins=20)
plt.show()
# %%[markdown]
# Okay!, so we don't even see anything here.
# %%
sns.histplot(x='Rating Count', data=df_clean[df_clean['Rating Count']<42], bins=20)
plt.show()
# This shows that majority of apps don't even get any ratings.
# And just a few get over hundreds and thousands of ratings.
# %%
# The maximum value in the above plot show 1e6, i.e., a million.
# Let's see how many apps have over a million ratings
len(df_clean[df_clean['Rating Count']>1e6])
# %%[markdown]
# Just 829 of 2 million+ apps have over a million ratings.

# %%
# Let's try to see apps from which category have higher ratings,
# and which are the categories that get rated the most.
df_clean[df_clean['Rating']>3.5]['Category'].value_counts().head().plot.barh()
# So the education category has the higest rated apps.
# %%
df_clean[df_clean['Rating Count']>1e6]['Category'].value_counts().head(6).plot.barh()
# Action apps have the most number of ratings. 
# 74 of total apps with more than a million reviews belong to action category. These could be the action games which are super popular.
# Sports and music have the same number of ratings and are in top 5.
# Tools category also has 54 apps over a million reviews. These could be the productivity tool apps that many people use on a regular basis.
# Lets just see some of these apps.
# %%
df_clean[df_clean['Rating Count']>1e6][df_clean['Category']=='Action'][['App Name', 'Rating', 'Rating Count']].head(10)
# As we suspected, it's the most popular action games, such as Shadow Fight 2, PUBG, Among Us, etc.

# %%
df_clean['Minimum Installs'].value_counts(normalize=True).plot.barh()
# On checking the install count, we see that the majority of apps fall 
# in the install range of 10 to 10,000.
# %%
df_clean['Maximum Installs'].describe()
# We don't really get much information from this. 

# %%
#Handling : 
#Relased col, Privacy policy, ad-supported, in app purchases, editors choice and scraped time
# %%
#released column
sample_values1 = df_clean['Released'].sample(n=20)
print(sample_values1)

# %%
# Imputing the missing values of released column  with median date 
df_clean['Released'] = pd.to_datetime(df['Released'], errors='coerce')
median_date = df_clean['Released'].median()  # Calculate the median date
df_clean['Released'].fillna(median_date, inplace=True)
df_clean['Released'].isna().sum() #recheking for missing values

# %%
#Calculating age of the app, by extracting the release date from the current date

df_clean['Year Released']= df_clean['Released'].dt.year #extracting year, month and day
df_clean['Month Released']= df_clean['Released'].dt.month
df['Day of week Released']= df_clean['Released'].dt.dayofweek

current_date=pd.to_datetime('now')
df_clean['App Age'] = round((current_date - df_clean['Released']).dt.days / 365.25,2) if pd.__version__ >= '1.1.0' else (current_date - df['Released']).days / 365.25
#print(df['App Age'])
# %%
#visualization of released column
#exploring distribution of app over the ages

plt.figure(figsize=(10,6))
sns.countplot(x= 'Year Released', data= df_clean,hue='Year Released', legend=False, palette = 'viridis')
plt.title('Distribution of App Releases Over the Years')
plt.xlabel('Year Released')
plt.ylabel('Number of Apps')
plt.show()

#Line plot with aggregated counts
plt.figure(figsize=(12, 6))
df_clean['Year Released'].value_counts().sort_index().plot(kind='line', marker='o', color='skyblue')
plt.title('Trend of App Releases Over the Years')
plt.xlabel('Year Released')
plt.ylabel('Number of Apps')
plt.show()

# %%
#Privacy column
#Handling missing values
df_clean['Privacy Policy'].isnull().sum()
df_clean['Privacy Policy'].fillna('Not Available', inplace = True) #Imputing missing value

# %%
#Creating a binary feature indicating whether the app has a privacy policy or not
df_clean['Has_PrivacyPolicy']= df_clean['Privacy Policy'].apply(lambda x: 1 if x != 'Not Available' else 0)
df_clean['Has_PrivacyPolicy']

# %%
#visualizing distribution of apps with and without privacy policy 

counts = df_clean['Has_PrivacyPolicy'].value_counts()

plt.figure(figsize=(8, 5))
counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Apps with and without Privacy Policies')
plt.xlabel('Has Privacy Policy')
plt.ylabel('Number of Apps')
plt.xticks(rotation=0)  
plt.show()

# %%
#df['Ad Supported'].head(10)
ad_support_counts = df_clean['IsAdSupported'].value_counts()
ad_support_counts
# %%
#Ad-supported Column
df_clean['Ad Supported'].isnull().sum()
df_clean['IsAdSupported'] = df_clean['Ad Supported']

# %%
#Visualization of number of apps ad supported vs not supported 
plt.figure(figsize=(8, 5))
sns.countplot(x='IsAdSupported', data=df_clean, palette='viridis',legend= False, hue= 'IsAdSupported')
plt.title('Distribution of Apps with and without Ad Support')
plt.xlabel('Is Ad Supported')
plt.ylabel('Number of Apps')
plt.show()

#Number of apps ad supported are almost the same as that not ad supported

# %%
df_clean['In App Purchases'].isnull().sum()
plt.figure(figsize=(8, 5))
sns.countplot(x='In App Purchases', data=df_clean, palette='viridis',hue= 'In App Purchases',legend= False)
plt.title('Distribution of Apps with and without In-App Purchases')
plt.xlabel('In App Purchases')
plt.ylabel('Number of Apps')
plt.show()

# %%
#Exploring Editor's Choice App

df_clean['Editors Choice'].isnull().sum()
df_clean['Editors Choice'].head(10)
Editor_counts = df_clean['Editors Choice'].value_counts()
print(Editor_counts)

# %%

#Visulization of the Editor's choice app

plt.figure(figsize=(8, 5))
sns.countplot(x='Editors Choice', data=df_clean, palette='viridis',legend= False, hue = 'Editors Choice')

plt.yscale('log') # Setting y-axis to log scale for better visualization if needed

plt.title("Distribution of Apps as Editor's Choice or Not")
plt.xlabel("Is Editor's Choice")
plt.ylabel('Number of Apps')
plt.show()

# %%
#Exploring Scraped Time column and visualization of the Scraped time column

df_clean['Scraped Time'].head(10)
df_clean['Scraped Time'].isnull().sum()
df_clean['Scraped Time'] = pd.to_datetime(df_clean['Scraped Time'], errors='coerce')

# Creating a time series plot of the number of apps scraped over time
plt.figure(figsize=(12, 6))
df_clean.groupby('Scraped Time').size().plot(title='Time Series of App Scraping', color='skyblue')
plt.xlabel('Scraped Time')
plt.ylabel('Number of Apps')
plt.show()

# %%
#Creating a time series chart to observe patterns in the scraping time over different time periods

df_clean['DayOfWeek'] = df_clean['Scraped Time'].dt.day_name() #Extracting day of the week and hour
df_clean['HourOfDay'] = df_clean['Scraped Time'].dt.hour
plt.figure(figsize=(14, 8))

# Time series plot for Day of the Week
plt.subplot(2, 1, 1)
sns.countplot(x='DayOfWeek', data=df_clean, palette='viridis', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],legend= False, hue= 'DayOfWeek')
plt.title('Number of Apps Scraped by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Apps')

# Time series plot for Hour of the Day
plt.subplot(2, 1, 2)
sns.countplot(x='HourOfDay', data=df_clean, palette='viridis', legend= False, hue= 'HourOfDay' )
plt.title('Number of Apps Scraped by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Apps')

plt.tight_layout()
plt.show()

# dev id, dev website, dev email, released, last updated, content rating
df_clean['Content Rating'].value_counts()
# %%
# Upon running the value counts function on the Content Rating column, it is observed that
# there are a total of six categories under which the apps have been sub-divided.
# The names of the categories seem to be a bit confusing ('Everyone'/ 'Everyone 10+'), so we'll provide better distinction to each.
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+', '17+')
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+', '10+')
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+', '18+')
# %%
df_clean['Content Rating'].value_counts(normalize=True).plot.barh()
# df_clean["Content Rating"]
# %%
df_clean['Last Updated'].head()
# %%
def splice_string(original_string, start, end=None):
    if end is None:
        return original_string[start:]
    else:
        return original_string[start:end]
# %%
df_clean['Year Last Updated'] = df_clean['Last Updated'].apply(lambda x: splice_string(x,8, ))
# %%
df_clean['Year Last Updated'] = df_clean['Year Last Updated'].astype(int)
# %%
df_clean['Year Last Updated'].max()
df_clean['Year Last Updated'].min()
# %%
sns.boxenplot(y="Year Last Updated", data=df_clean, palette="crest")
plt.ylabel("Year Last Updated")
# %%
# df_clean["Rating"].head(10)
# %%
sns.scatterplot(x='Content Rating', y='Rating', data=df_clean, palette="mako")
plt.title('Scatter Plot between Content Rating and Rating')
plt.ylabel('Rating')
plt.xlabel('Content Rating')
# %%
sns.stripplot(x='Year Last Updated', y='Rating', data=df_clean, palette="magma")
plt.title('Scatter Plot between Year of Last Update and Rating')
plt.ylabel('Rating')
plt.xlabel('Year Last Updated')
# %%
value_counts_table = df_clean['Year Last Updated'].value_counts().reset_index()
value_counts_table.columns = ['Year', 'Count']
table_str = tabulate(value_counts_table, headers='keys', tablefmt='pipe', showindex=False)
print(table_str)
#%%
df_clean['Year Last Updated'].value_counts().sort_index().plot(marker='o', color='#B28EC7')
plt.xlabel('Year Last Updated')
plt.ylabel('Number of Apps')
plt.title('App Update trend over the years')
plt.tight_layout()
# %%
df_clean['Developer Website'].isna().sum()
# %%
df_clean['has_developer_website'] = df_clean['Developer Website'].notna().astype(int)
# %%
sns.countplot(x="has_developer_website", data=df_clean, palette="PiYG")
plt.xlabel("Has Developer Website")
plt.ylabel("Number of Apps")
plt.title("Number of apps with or without Developer Website")
# %%
# sns.barplot(y='Rating', data=df_clean, hue='has_developer_website', palette="magma", color="yellow")
# plt.xlim(0,5)
# %%
plt.figure(figsize=(4, 8))
fig=px.treemap(df_clean, path=["Category", "Content Rating"], title="Count of Rating by age group by category")
fig.show()
# %%
fig = px.histogram(df_clean, x="Rating", color="has_developer_website",
                   marginal="violin", title="Number of apps by Rating, grouped by presence of developer website"
                  )
fig.show()
fig.update_layout(
    xaxis_title="Rating",
    yaxis_title="Count of Apps",
    width=750,  
    height=500
)
# %%

# fig = px.scatter(df_clean, x="Rating", y="Price", color="Content Rating",
#                   hover_data=['Category'])
# fig.show()
# fig.update_layout(
#     xaxis_title="Rating",
#     yaxis_title="Count of Apps",
# )

# %%
