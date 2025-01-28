#!/usr/bin/env python
# coding: utf-8

# #                               # REAL ESTATE PROJECT USING DATA SCIENCE 
# 

# In[ ]:





# #                              1. DATA CLEANING

# In[1]:


import pandas as pd
    data = pd.read_csv("C:\\Users\\sriha\\OneDrive\\Desktop\\pb excel\\Bengaluru_House_Data.csv")
df = pd.DataFrame(data)
df.head()


# In[2]:


df.shape


# In[3]:


df.area_type.unique()


# In[4]:


df.groupby('area_type')['area_type']. agg('count')


# In[5]:


df.isnull().sum()


# In[6]:


df1 = df.drop(['area_type','availability','society','balcony'] , axis='columns')


# In[7]:


df1.shape


# In[8]:


# checking the null values 
df1.isnull().sum()


# In[9]:


# now convert all  null values into MEAN or SD
df2 = df1.dropna()
df2.isnull().sum()


# In[10]:


df2.head()


# In[11]:


df2['size'].unique()


# In[12]:


# these are unique BHK for size,
# But, i want only integers and separate that integers

df2['BHK'] = df2['size']. apply (lambda x : int(x.split(' ')[0]))
df2.head()


# In[13]:


df2.shape


# In[14]:


df2.BHK.unique()


# In[15]:


df2[df2.BHK >20]


# In[ ]:





# In[16]:


# now seaperate the total_sqft
df2.total_sqft.unique()


# In[17]:


# let check how many values are '1133 - 1384' like between values 
def btwn_values(x):
    try:
        float(x)
    except:
        return False
    return True

df2[~(df2['total_sqft'].apply(btwn_values))]


# In[18]:


df2[~(df2['total_sqft'].apply(btwn_values))].shape


# In[19]:


#  Now these values to convert into AVERAGE values 
#  Like (x+y)/2 
# ex : (2100 - 2850)/ 2  ==> 4950/2  ==> 2475


# In[20]:


def convert_btwn_values(x):
    splits = x.split ('-')
    if len (splits)== 2:
        return (float(splits[0]) + float(splits[1])) / 2
    try:
        return float(x)
    except :
        return None


# In[21]:


convert_btwn_values('2600')


# In[22]:


convert_btwn_values('2100 - 2850')  


# In[23]:


#  now applying this to all the Table


# In[24]:


df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'] . apply( convert_btwn_values)
df3.head()


# In[25]:


df3.loc[122]


# In[26]:


df2['total_sqft'].unique()


# In[27]:


df3['total_sqft'].unique()  #  When compared to DF2  and DF3, the btwn values are ther change to Avarege values 


# In[ ]:





# #  2.  Future Engineering

# In[28]:


'''
    1. To find price_Per_sqft.
    2. Location no:of places 
'''


# In[29]:


df3.head()


# In[30]:


df4 = df3.copy()
df4['price_per_sqft'] = df4['price'] / df4['total_sqft']
df4.head()


# In[31]:


df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']    # Price values in Lakhs 
df4.head()


# In[32]:


df4['location'] .unique()


# In[33]:


len(df4['location'].unique())


# In[34]:


df4.location = df4.location.apply (lambda x: x.strip ())

location_status = df4.groupby('location')['location'] .agg ('count'). sort_values (ascending = False)
location_status


# In[35]:


len (location_status[location_status<= 10 ])


# In[36]:


location_less_than_10 = location_status[location_status<= 10 ]
location_less_than_10


# In[37]:


df4['location'] =  df4.location.apply(lambda x : '1-10 floors apartments' if x in location_less_than_10 else x)
df4.head(30)


# In[38]:


# Here Comes the houses and apartments are ehich is less than 10 floors apartment location are written as '1-10 floors apartment'


# In[ ]:





# # 3. Outliers Removal

# In[39]:


'''
        1. finding total_sqft/BHK 
        2.
        

'''


# In[40]:


df5 = df4.copy()
df5.head(10)


# In[41]:


df5['total_sqft/BHKs'] = df5['total_sqft'] / df5['BHK']  # finding price_per_BHK
df5.head()


# In[42]:


#  I want to separate below '300' price_per_BHKs

(df5[df5.total_sqft/ df5.BHK < 300])


# In[43]:


# that means 300 Price_per_BHKs are total  744 are there,
   # so, I want to remove this.
   
   
df5.shape   # Before remove


# In[44]:


df6 = df5[~(df5.total_sqft/ df5.BHK < 300)]
df6.shape   # After remove


# In[45]:


# removing below 300 price_per_BHKs
13246 - 12502


# In[46]:


df6.head()


# In[47]:


df6.drop('total_sqft/BHKs' , axis ="columns", inplace=True)


# In[48]:


df6.head()


# In[49]:


df6.describe()


# In[50]:


df6.price_per_sqft.describe()


# In[51]:


# According to Middle class they cam't buy 'Low' and 'Very Much High' , so all are looking middle range of rents 
# That way I want to remove all  'min' and 'Max' values.
#  so, I want  to remove the all less  MEAN and SD values.

import numpy as np
def remove_outliera(df):
    df_out = pd.DataFrame()
    for key,sub_df in df.groupby('location'):
        mean = np.mean(sub_df.price_per_sqft)
        std  = np.std(sub_df.price_per_sqft)
        reduced_df = sub_df[(sub_df.price_per_sqft > (mean - std))   & (sub_df.price_per_sqft <=(mean+std))]
        df_out = pd.concat([reduced_df,df_out], ignore_index=True)
    return df_out

df7 = remove_outliera(df6)
df7.head()


# In[52]:


df6.shape  # before  removing 


# In[53]:


df7.shape  # After removing    nearly 2000 values are removed


# In[54]:


df7.location.unique()


#      #   now when compared to same BHK size is same but, the Price is also a different
# ''' example --> uttarhalli location ,the total_sqft of house is 2BHK pricer is 81,000 same location the total_sqft of house is 2BHK pricer is 1,21,000
# 
#                 The price is totaly different 
#                 
#                 
#                 Like that we find like similar cases in datasets
#                 
#                 
#                 By using Scatter diagrams we find what are simliar BHKs and Prices are the different

# In[55]:


import matplotlib.pyplot as pp
def plot_scatter(df,location):
    bhk_2 = df[(df.location== location)  & (df.BHK==2)]
    bhk_3 = df[(df.location== location)  & (df.BHK==3)]
    pp.scatter (bhk_2.total_sqft , bhk_2.price ,  color = 'blue', label = 'BHK_2')
    pp.scatter (bhk_3.total_sqft , bhk_3.price , color =  'red', label = 'BHK_3')
    pp.xlabel("total_sqft")
    pp.ylabel('price')
    pp.grid()
    pp.legend()

plot_scatter(df7,"Rajaji Nagar")


#  Now i want to remove all those things which are 2BHK and 3BHKs are the same in line
# ''' We should also remove properities where same location, the price of 3Bedrooms apartmentis less than 2 Bedroom Apartment (with same sqft).
# what we will do is for a given location, we build a dictonary of stats per BHK
# 
#         1( Bed room):{
#           mean   = 4000
#           std    = 2000
#           count  = 34}
#           
#         2 (Bed room) :{
#         mean = 4300
#         std  = 2300
#         count = 22
#         }
#     Now we can removethose 2BHK apartments those price_per_sqft  is less than mean price_per_sqft of 1 BHK apartment
#     
# 

# In[56]:


import numpy as np

def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for BHK, BHK_df in location_df.groupby("BHK"):
            bhk_stats[BHK] = {
                'mean': np.mean(BHK_df['price_per_sqft']),
                'std': np.std(BHK_df['price_per_sqft']),
                'count': BHK_df.shape[0]
            }

        for BHK, BHK_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(BHK - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, BHK_df[BHK_df['price_per_sqft'] < (stats['mean'])].index.values)

    return df.drop(exclude_indices, axis='index')


# In[57]:


df7.shape


# In[58]:


df8 = remove_bhk_outliers(df7)
df8.shape         # When compared to  df7 and df8 nearly 3000 like BHK values are removed.


# In[59]:


plot_scatter(df8,"Rajaji Nagar")


# In[60]:


plot_scatter(df7,'Banashankari')


# In[61]:


plot_scatter(df8,"Banashankari")


# In[62]:


# Total in cities how amnt total_sqft  and counts 

pp.hist(df8.price_per_sqft,rwidth=0.5)
pp.xlabel('price_per_sqft')
pp.ylabel('counts')
pp.grid()


# In[63]:


# Now Bathrooms
df8.head()


# In[64]:


df8.bath.unique()


# In[65]:


df8[df8.bath> 10]


# In[66]:


pp.hist (df8.bath , rwidth= 0.7)
pp.xlabel("Number of Bathrooms")
pp.ylabel("count")
pp.title("Numnber of Bathrooms in Benglore ")
pp.grid()
pp.show()


# In[67]:


df8[df8.bath>df8.BHK+2]


# In[68]:


#Now  I want to remove Bathrooms which is higher than BHK  

df9 = df8[df8.bath < df8.BHK+2]
df9.shape


# In[69]:


df8.shape  # nearly 100 values are reduced.


# In[70]:


# here now data cleaning and everthing over 
# next performing using Machine Learning 
# Before we removing Unnessary columns

df10 = df9.drop (['size',"price_per_sqft"] , axis = 'columns')
df10.shape


# In[71]:


df10.head()


# In[ ]:





# # Model Buliding

# # using Machine Learning 

# In[72]:


dummy = pd.get_dummies(df10.location)
dummy


# In[73]:


df11 = pd.concat([df10,dummy], axis ="columns")
df11.head()


# In[74]:


# Now here location names are there in columns , so, I want to remove Location Column


# In[75]:


df12 = df11.drop ("location" , axis='columns')
df12.head()


# In[76]:


df12.shape


# In[ ]:





# In[ ]:





# In[77]:


X= df12.drop('price', axis ='columns')
X.head()


# In[78]:


y = df12.price
y


# In[79]:


from sklearn. model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2)


# In[80]:


len(X_test)


# In[81]:


len(df12)


# In[82]:


len(y_train)


# In[83]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# In[84]:


'''ShuffleSplit = It is used for cross-validation, which is a technique for 
                    assessing the performance of a model by dividing the dataset into different training and testing sets.
                    It uses to Random Shuffle , Create Multiple Splits , Adjustable parameters'''

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv= ShuffleSplit(n_splits=10 , test_size=0.2)
cross_val_score (LinearRegression(), X,y , cv= cv)


# # Test the model for few properties

# In[87]:


# What other regression like Lasso ,and DecisionTreeRegressor and other various regressors.
#  As data scientist I want to find which Regressor is best score . 
#  For that we used method called 'GridSearchCV'

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model_gridsearchCV(X,y):
    algos ={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'Lasso':{
            'model': Lasso(),
            "params":{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'Decision_tree':{
            'model' :DecisionTreeRegressor(),
            'params' :{
                'criterion' :['mse','friedman_mse'],
                'splitter' :['best','random']
            }
        }
    }
    scores=[]
    cv= ShuffleSplit(n_splits=5, test_size=0.2)
    for algo_names, config in algos.items():
        gs = GridSearchCV(config['model'] , config['params'] , cv =cv , return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model' : algo_names,
            'best_score':gs.best_score_,
            "best_params":gs.best_params_
        })
    return pd.DataFrame(scores, columns =['model','best_score', 'best_params'])
find_best_model_gridsearchCV(X,y)


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

#   ''' Now the data we want check the predicted score for locations'''

# In[90]:


X.columns  # or  df12.columns


# # Test the model for few properties

# In[107]:


def predicted_price(location,total_sqft,bath,BHK):
    locat_index = np.where (X.columns ==location)[0][0]
    
    x  = np.zeros(len(X.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = BHK
    
    if locat_index>=0:
        x[locat_index] =1
        
    return lr.predict([x])[0]
 


# In[125]:


predicted_price("Whitefield", 1000, 2,2)


# In[109]:


predicted_price("Whitefield",1000,2,3)


# In[110]:


predicted_price('Whitefield',1000,3,3)


# In[112]:


max(df12.total_sqft)


# In[141]:


len(df10.location)


# In[148]:


len(df12.filter(regex='Whitefield', axis=1))


# In[ ]:





# For a comfortable living space for four members, 
# most experts recommend between 1,600 and 1,800 square feet, 
# with some suggesting that an ideal range could be between 1,800 and 2,000 square feet
# depending on individual needs and desired room sizes.

# In[126]:


predicted_price("Whitefield",1800, 2,3)


# In[113]:


predicted_price("Whitefield",1800, 2,2)


# In[114]:


predicted_price("Whitefield",1800, 3,3)


# In[127]:


predicted_price("Whitefield",1800, 3,2)

