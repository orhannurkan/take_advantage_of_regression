from math import cos, asin, sqrt, pi
import numpy as np
import pandas as pd
from sklearn.utils import resample


def distance(lat, lon, lat2, lon2, price):
    # print('GIRDI++++++1')
    result = pd.DataFrame([[np.sqrt((lat2[lat2.index[i]]-lat)**2+(lon2[lat2.index[i]]-lon)**2),
                            price[lat2.index[i]], lon2[lat2.index[i]], lat2[lat2.index[i]]] for i in range(len(lat2))])
    # print('GIRDI++++++2', lat2.index[0], lat2.index[1], lat2.index[2])
    # result = []
    # for i in range(len(list(lat2.index))):
    #     # print(np.sqrt((lat2[i]-lat)**2+(lon2[i]-lon)**2))
    #     # if()
    #     result.append(price[list(lat2.index)[i]])
    # np.sqrt((lat2[i]-lat)**2+(lon2[i]-lon)**2))
    # price[i], lon2[i], lat2[i]]
    # result = pd.DataFrame(result)
    # print('GIRDI++++++3')
    # [np.mean(pd.DataFrame(dist(l, df['longitude'][i], df['latitude'], df['longitude'], df['p_a'])).drop_duplicates(
    # ).sort_values(by=0).head(6)) for i, l in enumerate(df['latitude']) for ix, lx in enumerate(x_test['latitude']) if((lx == l) & (ix == idx))]
    # print(len(result))
    # print('GIRDI++++++4')
    # print('---===#################################',
    #   np.mean(result.drop_duplicates().sort_values(by=0).head(6))[1])
    # print('GIRDI++++++5')
    # print('---===', result[result.index((min(result[0])))])
    return np.mean(result.drop_duplicates().sort_values(by=0).head(6))[1]


def filter(df, tt):
    df['rooms_number+1'] = df['rooms_number']+1
    # I used a project profile report that explained by Sara yesterday it is useful
    # 1 - to use room numbers as a referans for the prediction I think we have to add 1 as value because
    # there is 0 values because of studios then we will see just at back-end room numbers one increased
    # df['facades_number'] = [int(x) if (x < 5) and (x > 0) else 0 for x in df["facades_number"]]
    # 2 - according to our old data set facades_number is trash
    # 3 - there is more detail relaterd to features but
    # the most dificult part is postcode then subtype then building state
    # I thought to test every features 1 by 1 and to see the effect to the score
    df["house_is"] = df["house_is"].astype(int)
    df["equipped_kitchen_has"] = df["equipped_kitchen_has"].astype(int)
    df["furnished"] = df["furnished"].astype(int)
    df["open_fire"] = df["open_fire"].astype(int)
    df["terrace"] = df["terrace"].astype(int)
    df["garden"] = df["garden"].astype(int)
    df["swimming_pool_has"] = df["swimming_pool_has"].astype(int)
    # df["building_state_agg"] = [1 if x == 'to_renovate' else (
    # 3 if x == 'good' else 2) for x in df["building_state_agg"]]
    print('*************************************************')
    # df = df.drop(['mean_postcode'], axis=1)
    df = df.drop(['source'], axis=1)
    df = df.drop(['price'], axis=1)
    df = df.drop(['property_subtype'], axis=1)
    # df = df.drop(['area'], axis=1)
    df = df.drop(['p_a'], axis=1)
    df = df.drop(['coeff'], axis=1)
    df = df.drop(['divisor'], axis=1)
    df = df.drop(['postcode'], axis=1)
    df = df.drop(['house_is'], axis=1)    # G:%0.5
    df = df.drop(['rooms_number'], axis=1)
    df = df.drop(['rooms_number+1'], axis=1)
    df = df.drop(['equipped_kitchen_has'], axis=1)
    df = df.drop(['furnished'], axis=1)  # G:%0.5
    # df = df.drop(['land_surface'], axis=1)
    df = df.drop(['open_fire'], axis=1)
    df = df.drop(['terrace'], axis=1)
    # df = df.drop(['terrace_area'], axis=1)
    df = df.drop(['garden'], axis=1)
    # df = df.drop(['garden_area'], axis=1)
    df = df.drop(['swimming_pool_has'], axis=1)  # G:%0.5
    df = df.drop(['building_state_agg'], axis=1)  # G:%0.1
    df = df.drop(['basement'], axis=1)  # G:%0.1
    # df = df.drop(['coeff'], axis=1)  # G:%0.1
    # df = df.drop(['fire'], axis=1)  # G:%0.1
    # df = df.drop(['pool'], axis=1)  # G:%0.1
    # df = df.drop(['furnish'], axis=1)  # G:%0.1
    # df = df.drop(['kitchen'], axis=1)  # G:%0.1
    # df = df.drop(['status'], axis=1)  # G:%0.1
    # df = df.drop(['longitude'], axis=1)  # DANGER DO not HIDE
    # df = df.drop(['latitude'], axis=1)  # DANGER DO not HIDE
    # df = df.drop(['region'], axis=1)
    # df = df.drop(['facades_number'], axis=1)  # 11224 (99.4%) missing values
    if tt == 2:
        df = df.drop(['price_area'], axis=1)  # G:%0.1
        df = df.drop(['fire'], axis=1)  # G:%0.1
        df = df.drop(['pool'], axis=1)  # G:%0.1
        df = df.drop(['furnish'], axis=1)  # G:%0.1
        df = df.drop(['kitchen'], axis=1)  # G:%0.1
        df = df.drop(['status'], axis=1)  # G:%0.1
        # df['p_a'] = np.zeros(len(df))
    df.info()
    # df = pd.get_dummies(df, prefix=['col1', 'col2'])
    return df


def distance2(lat, lon, lat2, lon2, price):
    p = pi/180
    print('. Working...')
    result = []
    for i in range(len(lat2)):
        a = 0.5 - cos((lat2[i]-lat)*p)/2 + cos(lat*p) * \
            cos(lat2[i]*p) * (1-cos((lon2[i]-lon)*p))/2
        result.append([12742 * asin(sqrt(a)), price[i]])
    return result


def p_a(df):
    df['coeff'] = [2 if df['land_surface'][i]+df['garden_area'][i] < 250 else (4 if df['land_surface'][i]+df['garden_area'][i] < 1000 else (
        8 if df['land_surface'][i]+df['garden_area'][i] < 5000 else (12 if df['land_surface'][i]+df['garden_area'][i] < 10000 else 16))) for i in range(len(df))]
    
    df['divisor'] = df['area'] + df['terrace_area'] + \
        df['garden_area'] / df['coeff'] + df['land_surface']/df['coeff']
    
    df['p_a'] = df['price']/df['divisor']
    
    f = ['open_fire', 'swimming_pool_has', 'furnished', 'equipped_kitchen_has']
    c = [-5000, -15000, -10000, -5000]
    for i in range(len(f)):
        df['p_a'] += np.where(df[f[i]] == True, c[i]/df['divisor'], 0)
    factors = ['AS_NEW', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_RESTORE']
    rate = [-600, -300, 300, 600]
    for i in range(len(factors)):
        df['p_a'] += np.where(df['building_state_agg'] == factors[i],
                              (rate[i]*(df['area'] + df['terrace_area'])/df['divisor']), 0)
    return df


def p_a_prediction(df, i, a):
    print('1-dogru mu a ===', a)
    coeff = [2 if df['land_surface'][i]+df['garden_area'][i] < 250 else (4 if df['land_surface'][i]+df['garden_area'][i] < 1000 else (
        8 if df['land_surface'][i]+df['garden_area'][i] < 5000 else (12 if df['land_surface'][i]+df['garden_area'][i] < 10000 else 16))) for i in range(len(df))]
    print('2-dogru mu coeff ===', coeff)
    divisor = df['area'][i] + df['terrace_area'][i] + \
        (df['garden_area'][i] + df['land_surface'][i])/coeff
    print('3-dogru mu divisor===', divisor)
    f = ['open_fire', 'swimming_pool_has', 'furnished', 'equipped_kitchen_has']
    c = [5000, 15000, 10000, 5000]
    for ix in range(len(f)):
        if df[f[ix]][i] == True:
            a += c[ix]/divisor
    print('4-dogru mu ===', a)
    factors = ['AS_NEW', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_RESTORE']
    rate = [600, 300, -300, -600]
    for ix in range(len(factors)):
        if df['building_state_agg'][i] == factors[ix]:
            a += rate[ix]*(df['area'][i] + df['terrace_area'][i])/divisor
    print('5-dogru mu ===', a)
    return a
