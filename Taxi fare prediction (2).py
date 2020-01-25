#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().system('pip install folium')


# In[3]:


import folium
import os
import path


# In[4]:


df=pd.read_csv("C:/Users/Rajshah/train1.csv",nrows = 1200000)
df.columns


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df[df['fare_amount']<0].head(10)


# In[11]:


df[df['fare_amount']<0].count()


# In[12]:


df = df.drop(df[df['fare_amount']<0].index, axis=0)


# In[13]:


print('New size: %d' % len(df))


# In[14]:


df[df['fare_amount']<0].count()


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


import seaborn as sns


# In[17]:


plt.scatter(x=df.fare_amount,y=df.index)
plt.ylabel('Index')
plt.xlabel('fare_amount')
plt.show()


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid") 
  
sns.boxplot(df.fare_amount, data = df)


# In[19]:


df[df.fare_amount<100].fare_amount.hist(bins=100,figsize=(15,8))


# In[20]:


df[(df.fare_amount>300)].count()


# In[21]:


df = df.drop(df[(df.fare_amount>300)].index, axis=0)


# In[22]:


print('New size: %d' % len(df))


# In[23]:


df['fare_amount'][(df.fare_amount<0) | (df.fare_amount>300)].count()


# In[24]:


base_map = folium.Map(location=[40.712776,-74.005974], control_scale=True, zoom_start=12)


# In[25]:


from folium.plugins import HeatMap


# In[26]:


df_m = df[['pickup_longitude','pickup_latitude']].values.tolist()
HeatMap(data=df_m, radius=10).add_to(base_map)


# base_map

# base_map.save('index.html')

# In[27]:


import datetime


# In[28]:


df.pickup_datetime = pd.to_datetime(df['pickup_datetime'], infer_datetime_format=True)


# In[29]:


df.pickup_datetime.head()


# In[30]:


df['year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Day'] = df['pickup_datetime'].dt.day
df['Day of Week'] = df['pickup_datetime'].dt.dayofweek
df['Hour'] = df['pickup_datetime'].dt.hour
df['Minute'] = df['pickup_datetime'].dt.minute


# In[31]:


df.head()


# In[32]:


df.tail()


# In[33]:


month_df = []
grouped = df.groupby('Month')
for i in range(1,12+1):
    a = grouped.get_group(i).copy()
    month_df.append(a[['pickup_latitude','pickup_longitude']].values.tolist())


# In[34]:


from folium.plugins import HeatMapWithTime


# In[35]:


HeatMapWithTime(month_df, radius=5).add_to(base_map)


# base_map

# In[36]:


print(df.pickup_longitude[1])
print(df.pickup_latitude[1])
print(df.dropoff_longitude[1])
print(df.dropoff_latitude[1])


# In[37]:


#coordinates = list(map('pickup_longitude','pickup_latitude'))
#coordinates
x = (40.711303,-74.016048)
y = (40.782004,-73.979268)
coordinates=[x,y]

map = folium.Map(location=[40.712776,-74.005974], zoom_start=11)

line=folium.PolyLine(locations=coordinates,weight=2,color = 'blue')
map.add_child(line)


# for each in df[:100].iterrows():
#     folium.CircleMarker([each[1]['pickup_latitude'],each[1]['pickup_longitude']],
#                         radius=3,
#                         color='blue',
#                         popup=str(each[1]['pickup_latitude'])+','+str(each[1]['pickup_longitude']),
#                         fill_color='#FD8A6C'
#                         ).add_to(base_map)
# base_map

# In[38]:


df['dropoff_latitude'].isnull().sum()


# In[39]:


df['dropoff_longitude'].isnull().sum()


# In[40]:


df[df.dropoff_longitude.isnull()==True].head(1)


# In[41]:


df[df.dropoff_latitude.isnull()==True].head(2)


# In[42]:


df.drop(df[df.dropoff_latitude.isnull()==True].index, axis=0, inplace=True)


# In[43]:


df.drop(df[df.dropoff_longitude.isnull()==True].index, axis=0, inplace=True)


# In[44]:


import math
from math import sin, cos, asin, sqrt, radians 


# In[45]:


def getDistance(lat1,lon1,lat2,lon2):
    r = 6373 # earth's radius
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = r*c
    
    return distance


# In[46]:


df['Distance']= getDistance(df.pickup_longitude,df.pickup_latitude,df.dropoff_longitude,df.dropoff_latitude)


# In[47]:


df['Distance'].head(3)


# In[48]:


df[df.Distance==0].shape


# In[49]:


import random


# In[50]:


idx = (df.Distance >= 0.05)
print('Old size: %d' % len(df))
df = df[idx]
print('New size: %d' % len(df))


# In[51]:


df.Distance.hist(bins=50, figsize=(12,4))
plt.xlabel('distance')
plt.title('Histogram')
df.Distance.describe()


# In[52]:


df['fare_per_km'] = df.fare_amount / df.Distance
df.fare_per_km.describe()


# In[53]:


# for difference in night taxi fare

idx = (df.Distance < 3) & (df.fare_amount < 100)
plt.scatter(df[idx].Distance, df[idx].fare_per_km)
plt.xlabel('distance mile')
plt.ylabel('fare per distance km')

theta = (16, 4.0)
x = np.linspace(0.1, 3, 50)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


# In[54]:


df.pivot_table('fare_per_km', index='Hour', columns='year').plot(figsize=(14,6))
plt.ylabel('Fare $USD / km');


# In[55]:


df['passenger_count'].unique()


# In[56]:


plt.scatter(x=df.passenger_count, y=df.index)
plt.xlabel('no of passenger')
plt.ylabel('index')
plt.show()


# In[57]:


df['passenger_count'][df.passenger_count==0].count()


# In[58]:


df = df.drop(df[df.passenger_count==0].index, axis=0)


# In[59]:


df = df.drop(df[df.passenger_count>10].index, axis=0)


# In[60]:


sns.boxplot(x=df.passenger_count, data=df)


# In[61]:


sns.boxplot(x=df.Day,y=df.fare_amount, data=df)


# In[62]:


plt.figsize=(20,15)
plt.hist(x=df.Hour, bins=75)
plt.xlabel('hour')
plt.ylabel('index')
plt.show()


# In[63]:


plt.figsize=(20,15)
plt.scatter(x=df.Hour,y=df.fare_amount, s=2)
plt.xlabel('Hour')
plt.ylabel('Fare_amount')
plt.show()


# In[64]:


plt.figsize=(20,15)
plt.scatter(x=df.passenger_count, y=df.index)
plt.xlabel('no. of passenger')
plt.ylabel('index')
plt.show()


# In[65]:


plt.figsize=(20,15)
plt.scatter(x=df.passenger_count, y=df.fare_amount)
plt.xlabel('no. of passenger')
plt.ylabel('fare_amount')
plt.show()


# In[66]:


plt.figsize=(20,15)
plt.scatter(x=df.Distance, y=df.fare_amount)
plt.xlabel('distance in radians')
plt.ylabel('fare_amount')
plt.show()


# In[67]:


df.groupby('passenger_count')['Distance', 'fare_amount'].mean()


# In[68]:


df_test = pd.read_csv('C:/Users/Rajshah/test.csv')
df_test.head()


# In[69]:


df_test.columns


# In[70]:


df_test.tail()


# In[71]:


long_min= min(df_test.pickup_longitude.min(),df_test.dropoff_longitude.min())
long_max = max(df_test.pickup_longitude.max(),df_test.dropoff_longitude.max())
print(long_min,',',long_max)


# In[72]:


lat_min =min(df_test.pickup_latitude.min(),df_test.dropoff_latitude.min())
lat_max = max(df_test.pickup_latitude.max(),df_test.dropoff_latitude.max())
print(lat_min,',',lat_max)


# In[73]:


df_test.describe()


# In[74]:


df_test.dtypes


# In[75]:


df_test.pickup_datetime = pd.to_datetime(df_test['pickup_datetime'], infer_datetime_format=True)


# In[76]:


df_test['year'] = df_test['pickup_datetime'].dt.year
df_test['Month'] = df_test['pickup_datetime'].dt.month
df_test['Day'] = df_test['pickup_datetime'].dt.day
df_test['Day of Week'] = df_test['pickup_datetime'].dt.dayofweek
df_test['Hour'] = df_test['pickup_datetime'].dt.hour
df_test['Minute'] = df_test['pickup_datetime'].dt.minute


# In[77]:


df_test.head()


# In[78]:


df_test.isnull().sum()


# In[79]:


df_test['Distance']= getDistance(df_test.pickup_longitude,df_test.pickup_latitude,df_test.dropoff_longitude,df_test.dropoff_latitude)


# In[80]:


df_test['Distance'].describe()


# # Training Data

# In[115]:


print(df.shape)
train = df.drop(['pickup_datetime','fare_per_km'], axis=1)
print(train.shape)


# In[116]:


print(df_test.shape)
test = df_test.drop(['key','pickup_datetime'], axis=1)
print(test.shape)


# In[83]:


df.iloc[:,:].head(1)


# In[117]:


train.columns


# In[111]:


test.columns


# In[122]:


X_train  =train.iloc[:,train.columns!='fare_amount']
y_train = train['fare_amount'].values
x_test = test


# In[125]:


print(X_train.shape)
print(X_train.columns)
print(X_train.dtypes)


# In[127]:


print(y_train.shape)


# In[126]:


print(x_test.shape)
print(x_test.columns)
print(x_test.dtypes)


# In[128]:


from sklearn.ensemble import RandomForestRegressor


# In[129]:


rf = RandomForestRegressor()


# In[130]:


rf.fit(X_train, y_train)


# In[132]:


rf_pred = rf.predict(x_test)


# In[133]:


submission = pd.read_csv('C:/Users/Rajshah/sample_submission.csv')
submission['fare_amount'] = rf_pred
submission.to_csv = ('submission_Random.csv')
submission.head(10)


# In[ ]:


get_ipython().system('pip install xgboost')


# In[134]:


import xgboost as xgb


# In[135]:


dtrain = xgb.DMatrix(X_train,label=y_train)
dtest= xgb.DMatrix(x_test)


# In[136]:


print(len(x_test))


# In[137]:


dtrain


# In[138]:


params = {'max_depth':7,
         'eta':1,
         'slent':1,
         'objective':'reg:linear',
         'eval_metric':'rmse',
         'learning_rate':0.05
         }
num_rounds=50


# In[139]:


xb = xgb.train(params, dtrain,num_rounds)


# In[140]:


y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)


# In[141]:


print(len(y_pred_xgb))


# In[142]:


submission = pd.read_csv('C:/Users/Rajshah/sample_submission.csv')


# In[143]:


submission['fare_amount'] = y_pred_xgb
submission.to_csv = ('submission_xgb.csv')
submission.head(10)


# In[ ]:




