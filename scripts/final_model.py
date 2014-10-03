import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVR

train_data = pd.read_csv("../data/train.csv", header=0)
test_data = pd.read_csv("../data/test.csv", header=0)

print 'Data cleaning'
train_data['month'] = train_data['datetime'].str[5:7]
train_data['hour'] = train_data['datetime'].str[11:13]

train_target = train_data['count'].values

train_data = train_data.drop(['datetime','casual','registered','count'], axis=1)

train = train_data.values

test_data['month'] = test_data['datetime'].str[5:7]
test_data['hour'] = test_data['datetime'].str[11:13]
dates = test_data['datetime'].values

test_data = test_data.drop(['datetime'], axis=1)

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
forest = RandomForestRegressor(n_esimators=100)
forest = forest.fit(train,train_target)

print 'Predicting'
pred = forest.predict(test).astype(int)

print 'Writing file'
pred_file = open("../data/results.csv","wb")
open_csv = csv.writer(pred_file)
open_csv.writerow(["datetime","count"])
open_csv.writerows(zip(dates, pred))
pred_file.close()

print 'Done'
