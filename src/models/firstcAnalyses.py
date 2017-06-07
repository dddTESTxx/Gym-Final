
# coding: utf-8

# In[1]:

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


# In[2]:

print(digits.data)


# In[3]:

digits.target


# In[4]:

digits.images[0]


# In[5]:

from sklearn import svm


# In[6]:

clf = svm.SVC(gamma=0.001, C=100.)


# In[8]:

clf.fit(digits.data[:-1], digits.target[:-1])


# In[9]:

clf.predict(digits.data[-1:])


# In[10]:

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm


# In[14]:

gym = pd.read_csv('/Users/Dan/Downloads/Crowdedness gym/data.csv')


# In[17]:

list(gym)


# In[18]:

result = sm.ols(formula="number_people ~ is_weekend + temperature", data=gym).fit()


# In[19]:

print result.params


# In[20]:

print result.summary()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



