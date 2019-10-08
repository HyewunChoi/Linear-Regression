#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[10]:


data=read_csv('6-9月展期到期金额.csv',encoding='GBK')
data=data.dropna()
data


# In[13]:


sns.pairplot(data, x_vars=['<=1000','1000<a<=5000','5000<a<=10000','10000<a<=50000','>=50000'], y_vars='exceeding', size=7, aspect=0.8,kind='reg')
plt.show()


# In[14]:


data.corr()


# In[18]:


feature_cols = ['<=1000','1000<a<=5000','5000<a<=10000','10000<a<=50000','>=50000']
x = data[feature_cols]
x = data[['<=1000','1000<a<=5000','5000<a<=10000','10000<a<=50000','>=50000']]
x


# In[19]:


y = data['exceeding']
# equivalent command that works if there are no spaces in the column name
y = data.exceeding
y


# In[39]:


linreg = LinearRegression()
model=linreg.fit(x,y)


# In[50]:


b=linreg.intercept_
b


# In[49]:


a=linreg.coef_
a


# In[56]:


result=a[0]*data['<=1000']+a[1]*data['1000<a<=5000']+a[2]*data['5000<a<=10000']+a[3]*data['10000<a<=50000']+a[4]*data['>=50000']+b
result=round(result)
result


# In[59]:


data.insert(1,'result',result)


# In[60]:


fig, ax1 = plt.subplots() 
ax2 = ax1.twinx() 
ax1.plot(data['exceeding'],'g-') 
ax2.plot(data['result'],'r-')


# In[62]:


rmse=sp.sqrt(sp.mean((data['result']-data['exceeding'])**2))
rmse


# In[63]:


r2_score(data['result'],data['exceeding'])


# In[ ]:




