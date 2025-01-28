#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


dataset=pd.read_csv('loan_approval_dataset.csv')


# In[3]:


dataset


# In[4]:


dataset.describe(include='all')


# In[5]:


dataset[' education']=dataset[' education'].map({' Graduate':1,' Not Graduate':0})
dataset[' self_employed']=dataset[' self_employed'].map({' Yes':1,' No':0})
dataset[' loan_status']=dataset[' loan_status'].map({' Approved':1,' Rejected':0})


# In[6]:


dataset


# In[7]:


dataset.isnull().sum()


# In[8]:


#we dont need loan_id
data_clean=dataset.drop('loan_id',axis=1)
data_clean


# In[9]:


X=data_clean.iloc[:,:-1].values
X


# In[10]:


y=data_clean.iloc[:,-1].values


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[14]:


X_train.shape


# In[15]:


y_train.shape


# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


model=RandomForestClassifier()


# In[18]:


model.fit(X_train,y_train)


# In[19]:


model.score(X_test,y_test)


# In[20]:


pred=model.predict(X_test)


# In[21]:


pred


# In[22]:


y_test


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


print(confusion_matrix(y_test,pred))


# In[25]:


print(classification_report(y_test,pred))


# In[26]:


import pickle


# In[27]:


with open('model.pkl','wb') as f :
    pickle.dump(model,f)


# In[ ]:




