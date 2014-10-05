import pandas as pd
import numpy as np
import csv as csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

seed_value = 40

random.seed(seed_value)
np.random.seed(seed_value)

data = pd.read_csv("../data/train.csv", header=0)
rows = random.sample(data.index,30)
pred_casual = forest_casual.predict(test).astype(int)
pred_registered = forest_registered.predict(test).astype(int)

train_data = data.ix[rows]
test_data = data.drop(rows)

print 'Data cleaning'
train_data['month'] = train_data['datetime'].str[5:7]
train_data['day'] = train_data['datetime'].str[8:10]
train_data['hour'] = train_data['datetime'].str[11:13]

train_count = train_data['count'].values
train_casual = train_data['casual'].values
train_registered = train_data['registered'].values

train_data = train_data.drop(['datetime','casual','registered','count'], axis=1)

train = train_data.values

test_data['month'] = test_data['datetime'].str[5:7]
test_data['day'] = test_data['datetime'].str[8:10]
test_data['hour'] = test_data['datetime'].str[11:13]

test_count = test_data['count'].values
test_casual = test_data['casual'].values
test_registered = test_data['registered'].values

test_data = test_data.drop(['datetime','casual','registered','count'], axis=1)

test = test_data.values

#print 'Outliers removal'
#forest = RandomForestRegressor(n_estimators=100)
#forest = forest.fit(train,train_target)
#
#pred = forest.predict(train).astype(int)
#
#err_abs = 100*np.abs(train_target-pred)/train_target
#
#not_outliers = err_abs < np.percentile(err_abs,95)
#
#train_2 = train[not_outliers,:]
#train_target_2 = train_target[not_outliers]
#
#print 'Training'
#forest = RandomForestRegressor(n_estimators=100)
#forest = forest.fit(train_2,train_target_2)

print 'Training'
forest_casual = RandomForestRegressor(n_estimators=100)
forest_registered = RandomForestRegressor(n_estimators=100)
forest_count = RandomForestRegressor(n_estimators=100)

forest_casual = forest_casual.fit(train,train_casual)
forest_registered = forest_registered.fit(train, train_registered)
forest_count = forest_count.fit(train, train_count)

print 'Predicting'
pred_count = forest_count.predict(test).astype(int)
pred_casual = forest_casual.predict(test).astype(int)
pred_registered = forest_registered.predict(test).astype(int)
pred_sum = pred_casual + pred_registered

print 'count: ' + mean_squared_error(test_count,pred_count).astype(str)
print 'casual: ' + mean_squared_error(test_casual,pred_casual).astype(str)
print 'registered ' + mean_squared_error(test_registered,pred_registered).astype(str)
print 'sum: ' + mean_squared_error(test_count,pred_sum).astype(str)
