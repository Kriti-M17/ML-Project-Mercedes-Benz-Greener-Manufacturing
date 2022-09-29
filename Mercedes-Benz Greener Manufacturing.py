#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[67]:


pip install xgboost


# In[68]:


train_data = pd.read_csv("C:/Users/hp/OneDrive/Desktop/Simplilearn/Ml/test.csv")
test_data = pd.read_csv("C:/Users/hp/OneDrive/Desktop/Simplilearn/Ml/train.csv")
print(train_data.shape)
print(test_data.shape)


# In[70]:


for i in train_data.columns:
    data_type = train_data[i].dtype
    if data_type == 'object':
        print(i)


# In[71]:


##### If for any column(s), the variance is equal to zero, then you need to remove those variable(s).


# In[72]:


variance = pow(train_data.drop(columns={'ID'}).std(),2).to_dict()

null_cnt = 0
for key, value in variance.items():
    if(value==0):
        print('Name = ',key)
        null_cnt = null_cnt+1
print('No of columns which has zero variance = ',null_cnt)


# In[73]:


train_data = train_data.drop(columns={'X11','X93','X107','X233','X235','X268','X289','X290','X293','X297','X330','X347'})
train_data.shape


# In[74]:


train_data.isnull().sum().any()


# In[75]:


## Apply Label Encoder


# In[76]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[77]:


train_data_feature = train_data.drop(columns={'ID'})
train_data_target = train_data.ID
print(train_data_feature.shape)
print(train_data_target.shape)


# In[78]:


train_data_feature.describe(include='object')


# In[79]:


train_data_feature['X0'] = le.fit_transform(train_data_feature.X0)
train_data_feature['X1'] = le.fit_transform(train_data_feature.X1)
train_data_feature['X2'] = le.fit_transform(train_data_feature.X2)
train_data_feature['X3'] = le.fit_transform(train_data_feature.X3)
train_data_feature['X4'] = le.fit_transform(train_data_feature.X4)
train_data_feature['X5'] = le.fit_transform(train_data_feature.X5)
train_data_feature['X6'] = le.fit_transform(train_data_feature.X6)
train_data_feature['X8'] = le.fit_transform(train_data_feature.X8)


# In[80]:


#### Perform dimensionality reduction.


# In[81]:


print(train_data_feature.shape)
print(train_data_target.shape)


# In[82]:


from sklearn.decomposition import PCA
pca = PCA(n_components=.95)


# In[83]:


train_data_feature_trans = pca.fit_transform(train_data_feature)
print(train_data_feature_trans.shape)


# In[84]:


#### Predict your test_df values using XGBoost


# In[87]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[88]:


train_x,test_x,train_y,test_y = train_test_split(train_data_feature_trans,train_data_target,test_size=.3,random_state=7)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[ ]:


#### XGBoost's hyperparameters tuning manually


# In[89]:


xgb_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.4, max_depth = 10, alpha = 6, 
                           n_estimators = 20)
model = xgb_reg.fit(train_x,train_y)
print('RMSE = ',sqrt(mean_squared_error(model.predict(test_x),test_y)))


# In[90]:


pred_test_y = model.predict(test_x)

plt.figure(figsize=(10,5))

sns.distplot(test_y[test_y<160], color="skyblue", label="Actual value")
sns.distplot(pred_test_y[pred_test_y<160] , color="red", label="Predicted value")
plt.legend()

plt.tight_layout()


# In[ ]:


#### k-fold Cross Validation using XGBoost


# In[91]:


dmatrix_train = xgb.DMatrix(data=train_data_feature_trans,label=train_data_target)

params = {'objective':'reg:linear', 'colsample_bytree': 0.3, 'learning_rate': 0.3, 'max_depth': 5, 'alpha': 10}

model_cv = xgb.cv(dtrain=dmatrix_train, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, 
                      metrics="rmse", as_pandas=True, seed=7)
model_cv.tail(4)


# In[ ]:


#### Prediction on test data set using XGBoost


# In[92]:


test_data = test_data.drop(columns={'X11','X93','X107','X233','X235','X268','X289','X290','X293','X297','X330','X347'})
test_data.shape


# In[94]:


test_data.isnull().sum().any()


# In[95]:


test_data_feature = test_data.drop(columns={'ID'})
print(test_data_feature.shape)


# In[96]:


test_data_feature.describe(include='object')


# In[97]:


test_data_feature['X0'] = le.fit_transform(test_data_feature.X0)
test_data_feature['X1'] = le.fit_transform(test_data_feature.X1)
test_data_feature['X2'] = le.fit_transform(test_data_feature.X2)
test_data_feature['X3'] = le.fit_transform(test_data_feature.X3)
test_data_feature['X4'] = le.fit_transform(test_data_feature.X4)
test_data_feature['X5'] = le.fit_transform(test_data_feature.X5)
test_data_feature['X6'] = le.fit_transform(test_data_feature.X6)
test_data_feature['X8'] = le.fit_transform(test_data_feature.X8)


# In[98]:


pca.fit(test_data_feature)


# In[99]:


test_data_feature_trans = pca.fit_transform(test_data_feature)
print(test_data_feature_trans.shape)


# In[100]:


test_pred = model.predict(test_data_feature_trans)
test_pred


# In[101]:


fig, ax = plt.subplots(1,2, figsize=(14,5))

train_plot = sns.distplot(train_data_target[train_data_target<200], bins=100, kde=True, ax=ax[0])
train_plot.set_xlabel('Target(train_data)', weight='bold', size=15)
train_plot.set_ylabel('Distribution', weight='bold', size=15)
train_plot.set_title(' Dist. of target for train data', weight='bold', size=15)

test_plot = sns.distplot(test_pred[test_pred<200], bins=100, kde=True, ax=ax[1])
test_plot.set_xlabel('Target(test_data)', weight='bold', size=15)
test_plot.set_ylabel('Distribution', weight='bold', size=15)
test_plot.set_title(' Dist. of target for test data', weight='bold', size=15)

plt.tight_layout()


# In[ ]:




