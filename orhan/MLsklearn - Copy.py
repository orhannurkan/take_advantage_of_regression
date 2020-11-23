<<<<<<< HEAD
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from filter import filter as fl
from filter import distance as dist
from filter import p_a as price_area
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

degree = 2
poly = PolynomialFeatures(degree)
linreg = LinearRegression()
steps = [('PolynomialFeatures', poly), ('LinearRegression', linreg)]
pipeline = Pipeline(steps)
# df = pd.read_csv('dataSet-full-old-new-yes-gps-cleared.csv')
df = pd.read_csv('x_train.csv')
df2 = pd.read_csv('x_test.csv')
df3 = pd.read_csv('y_test.csv')
fields = ['source', 'postcode', 'house_is', 'property_subtype', 'price', 'rooms_number', 'area', 'equipped_kitchen_has'
          'furnished', 'open_fire', 'terrace', 'terrace_area', 'garden', 'garden_area', 'land_surface', 'facades_number',
          'swimming_pool_has', 'region', 'building_state_agg', 'price_area']
df = price_area(df)  # Calculate
# pks = df['postcode'].unique()
# print('start=', len(pks), len(df))
# # df = pd.DataFrame(columns=['A'])
# ds_all = df[df['postcode'] == pks[0]].sort_values(by='p_a')
# print('ds_all=', ds_all)
# for pk in range(len(pks)):
#     ds = df[df['postcode'] == pks[pk]].sort_values(by='p_a')
#     # print(len(ds))
#     ds = ds.head(int(len(ds)*17/18))
#     ds = ds.tail(int(len(ds)*18/19))
#     # ds_all.join(ds)
#     # ds_all = pd.concat(ds_all, ds)
#     ds_all = ds_all.append(ds, ignore_index=True)
# df = ds_all
y = df['price']
y2 = df3['price']
df = fl(df, 2)
df2['p_a'] = np.zeros(len(df2))
df2['divisor'] = np.zeros(len(df2))
df2['coeff'] = np.zeros(len(df2))
df2 = fl(df2, 2)
X = df
X2 = df2
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)
reg = LinearRegression()
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test))
clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
pipeline.fit(x_train, y_train)
print(pipeline.score(x_test, y_test))
print('-----------real test-------------')
print(reg.score(X2, y2))
print(clf.score(X2, y2))
print(pipeline.score(X2, y2))
=======
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from filter import filter as fl
from filter import distance as dist
from filter import p_a as price_area
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

degree = 2
poly = PolynomialFeatures(degree)
linreg = LinearRegression()
steps = [('PolynomialFeatures', poly), ('LinearRegression', linreg)]
pipeline = Pipeline(steps)
# df = pd.read_csv('dataSet-full-old-new-yes-gps-cleared.csv')
df = pd.read_csv('x_train.csv')
df2 = pd.read_csv('x_test.csv')
df3 = pd.read_csv('y_test.csv')
fields = ['source', 'postcode', 'house_is', 'property_subtype', 'price', 'rooms_number', 'area', 'equipped_kitchen_has'
          'furnished', 'open_fire', 'terrace', 'terrace_area', 'garden', 'garden_area', 'land_surface', 'facades_number',
          'swimming_pool_has', 'region', 'building_state_agg', 'price_area']
df = price_area(df)  # Calculate
# pks = df['postcode'].unique()
# print('start=', len(pks), len(df))
# # df = pd.DataFrame(columns=['A'])
# ds_all = df[df['postcode'] == pks[0]].sort_values(by='p_a')
# print('ds_all=', ds_all)
# for pk in range(len(pks)):
#     ds = df[df['postcode'] == pks[pk]].sort_values(by='p_a')
#     # print(len(ds))
#     ds = ds.head(int(len(ds)*17/18))
#     ds = ds.tail(int(len(ds)*18/19))
#     # ds_all.join(ds)
#     # ds_all = pd.concat(ds_all, ds)
#     ds_all = ds_all.append(ds, ignore_index=True)
# df = ds_all
y = df['price']
y2 = df3['price']
df = fl(df, 2)
df2['p_a'] = np.zeros(len(df2))
df2['divisor'] = np.zeros(len(df2))
df2['coeff'] = np.zeros(len(df2))
df2 = fl(df2, 2)
X = df
X2 = df2
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)
reg = LinearRegression()
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test))
clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
pipeline.fit(x_train, y_train)
print(pipeline.score(x_test, y_test))
print('-----------real test-------------')
print(reg.score(X2, y2))
print(clf.score(X2, y2))
print(pipeline.score(X2, y2))
>>>>>>> 9ca8eb23111fa3e5cbda5d5b1b892417568819e3
