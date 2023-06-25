#!/usr/bin/env python
# coding: utf-8

# Erin Kane <br>
# MSDS670 Data Visualization <br>
# Koenig <br>
# 25 June 2023

# Using the superstore.csv dataset which features Orders, ship dates, customer id, region, category, etc... to create visuals that display in histogram, heatmap, and choropleth models.

# In[1]:


##import necessary packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
import itertools
warnings.filterwarnings(action='ignore')
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from pylab import rcParams
import statsmodels.api as sm


# In[5]:


df = pd.read_excel('Sample - Superstore.xlsx')


# In[6]:


df.info()


# In[7]:


df.head()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


df.isnull().sum()


# In[13]:


#dropping postal code data, this won't distrupt the dataset too much. 
df=df.dropna(axis=0)


# In[14]:


df.isnull().sum() #clean!


# **And now to create visuals!**

# In[31]:


corrmat=df.corr()
top_corr=corrmat.index
plt.figure(figsize=(5,5))
#plot the heatmap
g=sns.heatmap(df[top_corr].corr(),annot=True,cmap='crest')


# In[32]:


df.drop(['Row ID','Ship Date','Ship Mode','Customer ID','Postal Code','Order ID','Profit','Discount'],axis=1,inplace=True) 


# In[33]:


## Downloading this pre-processed data into train1.csv
df.to_csv('train1.csv')


# In[34]:


print(df.shape)
df.head() #new shape


# Analysis questions:- <br>
# which state of US has higher frequency of sales? <br>
# in which year we had max sales? <br>
# in which month we have max sales? <br>
# which days of the month yield higher sales? <br>
# what are the top 10 states with high sales? <br>
# what are the top 20 cities with high sales? <br>
# top products highly in demand? <br>
# Most frequent customers?

# In[35]:


## Determining the categories in the Country column of the dataframe
df['Country'].unique()


# In[36]:


## Determining the number of States in US
states=df['State'].unique()
np.count_nonzero(states)


# In[37]:


## Determining the number of Cities in US
cities=df['City'].unique()
np.count_nonzero(cities)


# In[38]:


#Q What are the top 20 cities in US with high sales count
top_cities= df['City'].value_counts().nlargest(15)
top_cities


# **New York City has the highest Sales**

# Q. Who are the most frequent customers?

# In[42]:


#Most frequent customers
top_customers= df['Customer Name'].value_counts().nlargest(15)
top_customers


# **William Brown is the top customer, most frequent**

# In[44]:


rslt_df = df[df['Customer Name'] == 'William Brown'] 
rslt_df.head()


# checking out the category data

# In[45]:


# determining the unique values of category column.
category=df['Category'].unique()
print(category)
print(np.count_nonzero(category))


# In[49]:


plt.rcParams['figure.figsize'] = (10, 8)
sns.barplot(x = df['Category'], y = df['Sales'], palette ='icefire')
plt.title('The Distribution of Sales in each Category', fontsize = 10)
plt.xlabel('Category', fontsize = 10)
plt.ylabel('Count', fontsize = 15)


# Q. What are the top products with high sales?

# In[50]:


# determining the total count of sub-categories/ products in the Supermarket Store
subcategory=df['Sub-Category'].unique()
print(subcategory)
print(np.count_nonzero(subcategory))
#There are 17 products/ sub-categories.


# In[53]:


# visualizing sub-category wise distribution of sales
plt.rcParams['figure.figsize'] = (19, 8)
sns.barplot(x = df['Sub-Category'], y = df['Sales'], palette ='icefire')
plt.title('The Distribution of Sales in each Sub-Category', fontsize = 30)
plt.xlabel('Sub-Category', fontsize = 20)
plt.ylabel('Count', fontsize = 20)


# In[55]:


#top 5 products highly in demand
top_products= df['Sub-Category'].value_counts().nlargest(5)
top_products


# In[56]:


# determining segments of customers
segment=df['Segment'].unique()
print(segment)
print(np.count_nonzero(segment))


# In[57]:


# visualizing Segment wise distribution of sales
plt.rcParams['figure.figsize'] = (19, 8)
sns.barplot(x = df['Segment'], y = df['Sales'], palette ='icefire')
plt.title('The Distribution of Sales in each Segment', fontsize = 20)
plt.xlabel('Segment', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# Q. What are top 10 states with high sales?

# In[62]:


#visualizing state-wise sales distribution
df.groupby(['State'])['Sales'].nunique().plot.bar(figsize = (19, 7), cmap= 'crest')
plt.gcf().autofmt_xdate()
plt.title('Comparing statewise sales frequency', fontsize = 30)
plt.xlabel('States in US', fontsize = 10)
plt.ylabel('Sales Frequency')
plt.show()


# In[64]:


#top 10 states with high sales
top_states= df['State'].value_counts().nlargest(10)
top_states


# In[65]:


print(df['State'].max()) # california is with high frequency sales whereas Wyoming has the overall maximum sale price.


# In[67]:


plt.rcParams['figure.figsize'] = (15, 8)
sns.distplot(df['Sales'], color = 'blue')
plt.title('The Distribution of Sales', fontsize = 30)
plt.xlabel('Range of Sales', fontsize = 15)
plt.ylabel('No. of Sales count', fontsize = 15)
plt.show()


# In[69]:


df['Order Date'] = pd.to_datetime(df['Order Date'], errors = 'coerce') # it was already datetime object before, not a necessary step


# In[70]:


#extracting Year out of the Date to do year-wise analysis
df['Year'] = df['Order Date'].dt.year
#extracting month out of the Date to do month-wise analysis
df['Month'] = df['Order Date'].dt.month
#extracting Day out of the Date to do daywise analysis
df['Date'] = df['Order Date'].dt.day
df.columns


# In[71]:


# separating dependent and independent featurea
X=df.copy()
X.drop(['Sales'],axis=1,inplace=True)
X.head() # independent features


# In[72]:


y=df.iloc[:,11] # target as well as dependent feature
y.head()


# Q. which year had max sales?

# In[73]:


## visualizing through boxplot
plt.rcParams['figure.figsize'] = (19, 8)
sns.boxplot(x = df['Year'], y = df['Sales'], palette ='icefire')
plt.title('The Distribution of Sales in each Year', fontsize = 30)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Sales Price', fontsize = 15)


# Q. Which months of the year yield highest sale price?

# In[81]:


# visualizing month-wise sales distribution
plt.rcParams['figure.figsize'] = (19, 8)
sns.barplot(x = df['Month'], y = df['Sales'], palette ='ch:s=-.2,r=.6')
plt.title('The Distribution of Sales in each month', fontsize = 30)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Sales', fontsize = 15)


# Highest sales were found to be in the month of March. <br>
# Maximum products were sold in the year beginning and end of the year; whereas the sales were consistent during mid of the year. <br>
# <br>
# **Q. Which days of the month yield higher sales?**

# In[82]:


#visualizing daywise sales distribution
plt.rcParams['figure.figsize'] = (19, 8)
sns.barplot(x = df['Date'], y = df['Sales'], palette ='colorblind')
plt.title('The Distribution of Sales in each day', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Sales', fontsize = 15)


# The Sales distribution throughout the month keeps varying; It is observed that it is mostly highest in the mid of the month. <br>
# <br>
# **Time Series Analysis of Sales and Order Date**

# In[83]:


# extracting them in separate dataframe
features=['Order Date','Sales']
salesplot=df[features]
salesplot.head()


# The Order Dates are not in sorted order, therefore sorting the dataframe according to date

# In[84]:


salesplot.sort_values(by='Order Date',inplace=True)
salesplot


# In[85]:


Order_date=salesplot['Order Date']
Sales=salesplot['Sales']

##Simple Scatter Plot
plt.plot_date(Order_date,Sales,xdate=True)
plt.gcf().autofmt_xdate()
plt.title('Sales Data')
plt.xlabel('Order Date')
plt.ylabel('Sales')


# Slide the above slider towards right or left to zoom in or zoom out the graph to view specific dates. Also at the top right corner hover the cursor to explore more options. You can also hover over the graph to view the SALES PRICE on specific date <br>
# <br>
# Sales Forcasting

# In[87]:


# loading the pre-processed data that we prepared in the EDA notebook- train1.csv
df1=pd.read_csv('train1.csv')
df1.head()


# In[88]:


df1.shape


# In[89]:


## for sales forecasting we only need Order Date and Sales coulmn of the train1.csv
features=['Order Date','Sales']
dfs=df1[features]
dfs.head()


# In[90]:


dfs.shape


# In[91]:


dfs.info()


# In[92]:


dfs.tail()


# In[93]:


#converting into datetime type
dfs['Order Date'] = pd.to_datetime(dfs['Order Date'], errors = 'coerce')
dfs.info()


# In[94]:


#setting index
dfs=dfs.groupby('Order Date')['Sales'].sum().reset_index()
dfs


# In[95]:


dfs=dfs.set_index('Order Date')
dfs.index


# In[97]:


#using start of each month as timestamp
y=dfs['Sales'].resample('MS').mean()
y['2015':]


# In[98]:


#visualising Sales Time Series Data
y.plot(figsize=(15,6))
plt.show()


# The pattern shows that overall sales goes down around beginning of the year. Also the growth rate of sale has eventually risen from 2015 to 2019. Sales were at peak near the end of 2018.
# <br>
# **Visualizing the Trends as well as seasonality in the time series data**

# In[102]:


rcParams['figure.figsize']=15,5
decomp=sm.tsa.seasonal_decompose(y,model='additive')
fig=decomp.plot()
plt.show()


# In[ ]:




