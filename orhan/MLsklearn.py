from pandas_profiling import ProfileReport
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter import filter as fl
from filter import distance as dist
from filter import p_a
from filter import p_a_prediction
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

# df.drop_duplicates()
# ##############1- this a,b,c checks 0 values and calculates price/area
# a = np.where(df['terrace_area'] == 0, True, False)
# b = np.where(df['garden_area'] == 0, True, False)
# c = np.where(df['land_surface'] == 0, True, False)
# df['price_area'] = df['price']/df['area']
# df['price_area'] = np.where(a * b * c, df['price']/df['area'], 0)
# following calculations are just for houses
# similarly we should find the calculation of price/area for terrace,garden and land
# -for example 100m2 house with terrace=0,garden=0 and land=0   get 1000 euros
# -same postcode 100m2 house with terrace = 10 maybe            get 1250 euros it means terrace m2 adds 25 euros
# -same postcode 100m2 house with garden = 10 maybe             get 1200 euros it means garden m2 adds 20 euros
# -same postcode 100m2 house with land = 10 maybe               get 1150 euros it means land m2 adds 15 euros
# so, we need to find relation between these 4 areas and ratio for that postcode
# and then we will use that values to calculate following columns and will store info instead of terrace,garden and land
# it means for prediction we will use these new columns(which will have the mean for that postcode) without knowing the price
# df['price_terrace_area'] = ??? --- 250
# df['price_garden_area'] = ???  --- 200
# df['price_land_surface'] = ??? --- 150
# ##############1- like mean of price/area more price/terrace_area, price/garden_area , price/land_area
# df2 = pd.read_csv('x_test.csv')
# df3 = pd.read_csv('y_test.csv')
# df2 = pd.read_csv('dataSet-full-old-new-yes-gps-cleared-test.csv')
# df['price_area'] = df.groupby('postcode')['price_area'].transform('mean')
fields = ['source', 'postcode', 'house_is', 'property_subtype', 'price', 'rooms_number', 'area', 'equipped_kitchen_has'
          'furnished', 'open_fire', 'terrace', 'terrace_area', 'garden', 'garden_area', 'land_surface', 'facades_number',
          'swimming_pool_has', 'region', 'building_state_agg', 'price_area']
# for i in range(len(df2)):#     if df2['land_surface'][i]+df2['garden'][i] < 250:#         coeff.append(2)
#     elif df2['land_surface'][i]+df2['garden'][i] < 1000:#         coeff.append(4)#     elif df2['land_surface'][i]+df2['garden'][i] < 5000:
#         coeff.append(8)#     elif df2['land_surface'][i]+df2['garden'][i] < 10000:#         coeff.append(12)#     else:#         coeff.append(16)
# -----
df = p_a(df)
# # ayni postcodu icin minimum %10 ve max %10 sil
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
#     # print('ds_all=', len(ds_all))
#     # print(len(ds))
# # pks = df['postcode'].unique()
# df = ds_all
# # print('finish=', len(pks), len(df))
# print('ds_all=', len(ds_all))
# -----
# print(len(ds), ds.tail(int(len(ds)/8)))
# print(df.groupby('postcode')['postcode'][pk])
# print(df.sort_values(by='p_a')[['p_a', 'postcode']].head(6))
# df2 = p_a(df2)
# df['p_a'] = df['p_a']+np.where(df['open_fire'] == True, 5000/divisor, 0)
# df['p_a'] = df['p_a'] + np.where(df['swimming_pool_has'] == True, 15000/divisor, 0)
# df['p_a'] = df['p_a']+np.where(df['furnished'] == True, 10000/divisor, 0)
# df['p_a'] = df['p_a'] + np.where(df['equipped_kitchen_has'] == True, 10000/divisor, 0)
# df['price_area'] = np.where(df['swimming_pool_has'] == True, df['price_area']+(15000/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])
# df['price_area'] = np.where(df['furnished'] == True, df['price_area']+(10000/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])
# df['price_area'] = np.where(df['equipped_kitchen_has'] == True, df['price_area']+(5000/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])

# df['price_area'] = np.where(df['building_state_agg'] == "AS_NEW", df['price_area']+(600*(df['area'] + df['terrace_area'])/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])
# df['price_area'] = np.where(df['building_state_agg'] == "JUST_RENOVATED", df['price_area']+(300*(df['area'] + df['terrace_area'])/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])
# df['price_area'] = np.where(df['building_state_agg'] == "TO_RENOVATE", df['price_area']-(300*(df['area'] + df['terrace_area'])/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])
# df['price_area'] = np.where(df['building_state_agg'] == "TO_RESTORE", df['price_area']-(600*(df['area'] + df['terrace_area'])/(df['area'] + df['terrace_area'] +df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff'])), df['price_area'])

# RUN these 3 lines just one time for a dataset to get detailed info
# 1- from pandas_profiling import ProfileReport
# 2- profile = ProfileReport(df, title="ImmoEliza data")
# 3- profile.to_file("reports_ImmoEliza.html")

# use median instead of mean down tyr and see the resuls
# df['mean_postcode'] = df.groupby('postcode')['price_area'].transform('mean')
# df2['mean_postcode'] = df.groupby('postcode')['price_area'].transform('mean')
# df2['price_area'] = df.groupby('postcode')['price_area'].transform('mean')
# OUTLIERS-----------------------------
# Delete ROWS where roomnumber is > 7
# df = df[df['rooms_number'] < 8]
# df = df[df['price'] < 4000000]  # to see outlier how much effect the prediction
# df = df.drop(['land_surface'], axis=1)
# df = df.drop(['terrace_area'], axis=1)
# df = df.drop(['garden_area'], axis=1)
# for i in fields:
#     # print(i)
#     sns.boxplot(x=df[i])
#     # print(df.columns)
#     plt.show()
# OUTLIERS-----------------------------
# df = df[df['postcode'] == 9000]
# df2 = pd.read_csv('x_test.csv')
# df3 = pd.read_csv('y_test.csv')
y = df['price']
y2 = df3['price']
df = fl(df, 2)
df2['p_a'] = np.zeros(len(df2))
df2['divisor'] = np.zeros(len(df2))
df2['coeff'] = np.zeros(len(df2))
# X2['p_a'] = pd.DataFrame([
# dist(lx,
# X2['longitude']
# [X2.index[X2['latitude'] == lx][0]],
# df['latitude'], df['longitude'], df['p_a'])
#                           for ix, lx in enumerate(X2['latitude'])])
aaa = 0
p_a = []
with open('completed.csv', 'w') as file:
    for ix, lx in enumerate(df2['latitude']):
        aaa += 1
        print(aaa, '. kayit icin islem yapiliyor')
        a = dist(lx, df2['longitude'][df2.index[df2['latitude'] == lx][0]],
                 df['latitude'], df['longitude'], df['p_a'])
        # print('index = ', df2.index[df2['latitude'] == lx][0])
        i = df2.index[df2['latitude'] == lx][0]
        ###########################################################
        coeff = [2 if df2['land_surface'][i]+df2['garden_area'][i] < 250 else (4 if df2['land_surface'][i]+df2['garden_area'][i] < 1000 else (
            8 if df2['land_surface'][i]+df2['garden_area'][i] < 5000 else (12 if df2['land_surface'][i]+df2['garden_area'][i] < 10000 else 16))) for i in range(1)]
        # print('2-dogru mu coeff ===', coeff) coeffs are related to postcode ???????????????????????????????????????
        divisor = df2['area'][i] + df2['terrace_area'][i] + \
            (df2['garden_area'][i] + df2['land_surface'][i])/coeff[0]
        # print('3-dogru mu divisor===', divisor)
        f = ['open_fire', 'swimming_pool_has',
             'furnished', 'equipped_kitchen_has']
        c = [5000, 15000, 10000, 5000]
        for ix in range(len(f)):
            if df2[f[ix]][i] == True:
                a += c[ix]/divisor
        # print('4-dogru mu ===', a)
        factors = ['AS_NEW', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_RESTORE']
        rate = [600, 300, -300, -600]
        for ix in range(len(factors)):
            if df2['building_state_agg'][i] == factors[ix]:
                a += rate[ix]*(df2['area'][i] + df2['terrace_area'][i])/divisor
        print('FINAL a ===', a)
        ###########################################################
        p_a.append(a)
        file.write(str(a)+','+str(lx)+','+str(df2['longitude']
                                              [df2.index[df2['latitude'] == lx][0]])+'\n')
df2['p_a'] = pd.DataFrame(p_a)
df2 = fl(df2, 2)
# pd.get_dummies(df, prefix=['col1', 'col2'])
# print('---------------', df2.shape, df.shape)
# , 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'
# df = pd.get_dummies(df, prefix=['col1', 'col2'])
# df2 = pd.get_dummies(df2, prefix=[
#  'col1', 'col2'])  # , 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'
# print(df2.shape, df.shape)
X = df
X2 = df2
# x2_train, x2_test, y2_train, y2_test = train_test_split(
# X2, y2, test_size=0.20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)
# ---THIS IS THE LONGEST PROCESS---
# print(df2.shape, df.shape)
# print(X2.shape, X.shape)
# print(x2_train.shape, x_train.shape)
# print(y2.shape, y.shape)
# print(y2_train.shape, y_train.shape)
# df.to_csv(path_1+"/dataSet.csv", encoding='utf-8', index=ids)
# x_train.to_csv('x_train.csv')
# y_train.to_csv('y_train.csv')
# x_test.to_csv('x_test.csv')
# y_test.to_csv('y_test.csv')
# print(x_test['p_a'])  # , x_test['latitude'], x_test['longitude']
# print('=====================================')
# for idx in range(len(x_test['p_a'])):
#     print('***********')
#     # if((lx == l) & (ix == idx)) for i, l in enumerate(x_train['latitude'])
#     # (x_test['longitude'][x_test.index[x_test['latitude'] == lx][0]], lx)
#     # np.mean(pd.DataFrame(.drop_duplicates().sort_values(by=0).head(6)
#     print([dist(lx, x_test['longitude'][x_test.index[x_test['latitude'] == lx][0]], x_train['latitude'], x_train['longitude'], x_train['p_a'])
#            for ix, lx in enumerate(x_test['latitude'])])
# print(len(x_test), x_test['p_a'])
reg = LinearRegression()
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test))
clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
# print(clf.predict(x_test))
# poly ======================================
pipeline.fit(x_train, y_train)
print(pipeline.score(x_test, y_test))
# print(pipeline.predict(x_test))
print('-----------reel test-------------')
print(reg.score(X2, y2))  # print(reg.score(x2_train, y2_train))
print(clf.score(X2, y2))  # print(clf.score(x2_train, y2_train))
# print(pipeline.score(x2_train, y2_train))
print(pipeline.score(X2, y2))
# print(pipeline.predict(x2_train))
# print(y2_train)
print('-----------reel test-------------')
# pipeline.fit(np.array(x_train).reshape(-1, 1),
#              np.array(pd.DataFrame((np.array(y_train))).fillna(0)).reshape(-1, 1))
# poly ======================================
# print(x_test)
# print(y_test)
# # profile = ProfileReport(df, title="ImmoEliza data")
# # profile.to_file("reports_ImmoEliza.html")
