# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:59:06 2017

@author: vino
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Set path 
os.chdir('C:/Users/vino/Desktop/DengAI')

#import training datasets
deng_train_x = pd.read_csv('dengue_features_train.csv',header=0)
deng_train_y = pd.read_csv('dengue_labels_train.csv',header=0)
deng_train_x.head()
deng_train_y.head()

# Explore the data
deng_train_x.dtypes
deng_train_y.dtypes
deng_train_x['city'] = deng_train_x['city'].astype(str)
deng_train_x['week_start_date'] = pd.to_datetime(deng_train_x['week_start_date'],format='%Y-%m-%d')

# merge the dataframes
deng_train = pd.merge(left=deng_train_x,
                      right=deng_train_y,
                      on=['city','year','weekofyear'],
                      how='inner')        
##############handling Missing Values########################
# identify the columns having missing values
column_list = deng_train.isnull().any()
na_columns=column_list[column_list==True].index.values

# identify the number of missing values in each column
total_df_rows = deng_train.shape[0]
num_miss_values = total_df_rows-deng_train.count()
print(num_miss_values)

# isolate all the rows with missing values
miss_df = pd.DataFrame(columns=deng_train.columns.values)
for i in range(total_df_rows):
    if(deng_train.iloc[i,:].isnull().any()):
        miss_df = miss_df.append(deng_train.iloc[i,:])
    else:
        pass
miss_df.head()

# draw the density plot of the na columns
for i in na_columns:
    sns.kdeplot(deng_train[i].dropna())
    plt.title('Distribution of '+ str(i)+ ' column')
    plt.figure()

# we see all the columns are almost normally distributed. Hence we can replace the missing values with the column mean

# replace the missing values with their mean
for i in na_columns:
    deng_train[i][deng_train[i].isnull()==True] = float(deng_train[i].dropna().mean())

# check if any na values are present
deng_train.isnull().any()
#all the columns should be false
##############################################################

##################label encoding##############################
# Encode the 'city' column to numeric label
city_encode = LabelEncoder()
city_encode.fit(deng_train['city']) 
deng_train['city'] = city_encode.transform(deng_train['city'])
deng_train.head()
deng_train.tail()
##############################################################

##################understanding Correlation###################
# Correlation heat map
sns.heatmap(deng_train.corr(),cmap='seismic')
plt.yticks(rotation='horizontal',fontsize=7)
plt.xticks(rotation=75,fontsize=7)

#correlation matrix
corr_matrix = deng_train.corr()
corr_matrix['total_cases'].sort_values()

#flattened correlation matrix
a = []
for i in corr_matrix.index.values:
    for j in corr_matrix.columns.values:
        a.append(i+' + '+j)
corr_matrix_flat = pd.DataFrame(corr_matrix.values.flatten(),
                                columns = ['Values'],index=a)
corr_2 = corr_matrix_flat[corr_matrix_flat['Values']>=0.5]
##############################################################

###############check for outliers#############################
# removing unuseful column
x = deng_train.copy()
del x['week_start_date']

# drawing boxplots for all the predictors
for i in x.columns.values:
    plt.boxplot(x[i])
    plt.title('Box Plot of '+str(i)+' column')
    plt.figure()
##############################################################

##########predictor matrix and response term matrix###########
#Feature Matrix
x = deng_train.copy()
del x['week_start_date']
del x['total_cases']
x = x.as_matrix()
#Label Matrix
y = deng_train['total_cases'].as_matrix()
##############################################################

##############Test and Train data split#######################
#x_train,x_test,y_train,y_test = #train_test_split(x,y,train_size=0.80,random_state=777)
##############################################################

######### Random Forest Regression ##################
rf_regressor = RandomForestRegressor(n_estimators=75,
                                     criterion='mae',
                                  
                                     random_state=333)
model_fit = rf_regressor.fit(x,y)

#mean_absolute_error(y_train,a)

############Predicting Test observations########################
actual_test = pd.read_csv('dengue_features_test.csv',header=0)
actual_test['city']=city_encode.transform(actual_test['city'])
del actual_test['week_start_date']


column_list_test = actual_test.isnull().any()
na_columns_list=column_list_test[column_list_test==True].index.values
for i in na_columns_list:
    actual_test[i][actual_test[i].isnull()==True] = float(actual_test[i].dropna().mean())

actual_test = actual_test.as_matrix()

#################predicting values##################################
pred = rf_regressor.predict(actual_test).astype(int)
##############################################################







    





