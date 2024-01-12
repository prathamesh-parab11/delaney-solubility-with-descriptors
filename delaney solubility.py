#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv('delaney_solubility_with_descriptors.csv')
df


# In[5]:


y=df['logS']
y


# In[6]:


x=df.drop('logS',axis=1)
x


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[9]:


x_train


# In[10]:


x_test


# In[11]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[12]:


y_lr_train_pred=lr.predict(x_train)
y_lr_test_pred=lr.predict(x_test)
y_lr_train_pred
y_lr_test_pred


# In[13]:


from sklearn.metrics import mean_squared_error,r2_score


# In[14]:


lr_train_mse=mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2=r2_score(y_train,y_lr_train_pred)
lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2=r2_score(y_test,y_lr_test_pred)


# In[15]:


print("lr_train_mse : ",lr_train_mse)
print("lr_train_r2 : ",lr_train_r2)
print("lr_test_mse : ",lr_test_mse)
print("lr_test_r2 : ",lr_test_r2)


# In[16]:


lr_results=pd.DataFrame(['linear regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results


# In[17]:


lr_results.columns=['method','training mse','training r2','test mse','test r2']
lr_results


# In[18]:


from sklearn.ensemble import RandomForestRegressor


# In[19]:


rf=RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(x_train,y_train)


# In[20]:


y_rf_train_pred=rf.predict(x_train)
y_rf_test_pred=rf.predict(x_test)
y_rf_train_pred
y_rf_test_pred


# In[21]:


from sklearn.metrics import mean_squared_error,r2_score
rf_train_mse=mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2=r2_score(y_train,y_rf_train_pred)
rf_test_mse=mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2=r2_score(y_test,y_rf_test_pred)


# In[22]:


print("rf_train_mse : ",rf_train_mse)
print("rf_train_r2 : ",rf_train_r2)
print("rf_test_mse : ",rf_test_mse)
print("rf_test_r2 : ",rf_test_r2)


# In[23]:


rf_results=pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results


# In[24]:


rf_results.columns=['method','training mse','training r2','test mse','test r2']
rf_results


# In[25]:


df_models=pd.concat([lr_results,rf_results],axis=0)


# In[26]:


df_models.reset_index(drop=True)


# In[27]:


import matplotlib.pyplot as plt
import numpy as np


# In[29]:


plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train))
plt.ylabel('Predict Log S')
plt.xlabel('Experimental Log S')


# In[47]:


pip show matplotlib


# In[ ]:




