#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy as sp
from sklearn.metrics import r2_score


# In[34]:



data = read_csv('6-9月展期到期.csv',encoding='GBK')


# In[35]:


data


# In[36]:


data=data.dropna()


# In[37]:


data


# In[38]:


plt.scatter(data['到期总数'],data['展期数'])


# In[39]:


data.corr()


# In[40]:


lrModel = LinearRegression()


# In[41]:


x = data[['到期总数']]
y = data[['展期数']]


# In[42]:


lrModel.fit(x,y)


# In[43]:


lrModel.score(x,y)


# In[44]:


lrModel.predict([[10000],[70]])


# In[45]:


alpha = lrModel.intercept_[0]


# In[46]:


beta = lrModel.coef_[0][0]


# In[47]:


alpha + beta*numpy.array([7628,7467])


# In[48]:


alpha


# In[49]:


beta


# In[50]:


a=alpha + beta*data['到期总数']


# In[66]:


pd.Series(a)


# In[79]:


data


# In[85]:


fig, ax1 = plt.subplots() 
ax2 = ax1.twinx() 
ax1.plot(data['展期数'],'g-') 
ax2.plot(data['a'],'r-') 


# In[89]:


sns.regplot('到期总数', '展期数', data=data)


# In[91]:


predictions = lrModel.predict(x)


# In[93]:


plt.figure(figsize=(16, 8))
plt.scatter(
    data['到期总数'],
    data['展期数'],
    c='black'
)
plt.plot(
    data['到期总数'],
    predictions,
    c='blue',
    linewidth=2
)
plt.show()


# In[95]:


rmse=sp.sqrt(sp.mean((data['a']-data['展期数'])**2))


# In[96]:


rmse


# In[106]:


def get_r2_numpy(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r_squared = 1 - (sum((y - (slope * x + intercept))**2) / (len(y) * np.var(y)))
    return r_squared


# In[107]:


get_r2_numpy(data['到期总数'],data['展期数'])


# In[109]:


r2_score(data['展期数'],data['a'])


# In[ ]:




