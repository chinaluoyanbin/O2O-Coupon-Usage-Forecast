import pandas as pd
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings('ignore')

dataset1 = pd.read_csv('data/dataset1.csv', keep_default_na=False)
feature1 = pd.read_csv('data/feature1.csv', keep_default_na=False)
dataset2 = pd.read_csv('data/dataset2.csv', keep_default_na=False)
feature2 = pd.read_csv('data/feature2.csv', keep_default_na=False)
dataset3 = pd.read_csv('data/dataset3.csv', keep_default_na=False)
feature3 = pd.read_csv('data/feature3.csv', keep_default_na=False)

############# user related feature   #############
"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
      

"""


def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(
        int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


#for dataset3
user3 = feature3[[
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received', 'date'
]]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user3[user3.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][[
    'user_id', 'distance'
]]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user3[user3.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user3[user3.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user3[(user3.date_received != 'null') & (user3.date != 'null')][[
    'user_id', 'date_received', 'date'
]]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(
    get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(
    columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'},
    inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(
    columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'},
    inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(
    columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'},
    inplace=True)

user3_feature = pd.merge(t, t1, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t3, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t4, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t5, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t6, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t7, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t8, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t9, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t11, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t12, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t13, on='user_id', how='left')
user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan, 0)
user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan, 0)
user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype(
    'float') / user3_feature.buy_total.astype('float')
user3_feature[
    'user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype(
        'float') / user3_feature.coupon_received.astype('float')
user3_feature.buy_total = user3_feature.buy_total.replace(np.nan, 0)
user3_feature.coupon_received = user3_feature.coupon_received.replace(
    np.nan, 0)
user3_feature.to_csv('data/user3_feature.csv', index=None)

#for dataset2
user2 = feature2[[
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received', 'date'
]]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user2[user2.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][[
    'user_id', 'distance'
]]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user2[user2.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user2[user2.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user2[(user2.date_received != 'null') & (user2.date != 'null')][[
    'user_id', 'date_received', 'date'
]]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(
    get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(
    columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'},
    inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(
    columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'},
    inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(
    columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'},
    inplace=True)

user2_feature = pd.merge(t, t1, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t8, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t12, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t13, on='user_id', how='left')
user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan, 0)
user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan, 0)
user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype(
    'float') / user2_feature.buy_total.astype('float')
user2_feature[
    'user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype(
        'float') / user2_feature.coupon_received.astype('float')
user2_feature.buy_total = user2_feature.buy_total.replace(np.nan, 0)
user2_feature.coupon_received = user2_feature.coupon_received.replace(
    np.nan, 0)
user2_feature.to_csv('data/user2_feature.csv', index=None)

#for dataset1
user1 = feature1[[
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received', 'date'
]]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[user1.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][[
    'user_id', 'distance'
]]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user1[user1.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user1[user1.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user1[(user1.date_received != 'null') & (user1.date != 'null')][[
    'user_id', 'date_received', 'date'
]]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(
    get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(
    columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'},
    inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(
    columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'},
    inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(
    columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'},
    inplace=True)

user1_feature = pd.merge(t, t1, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t3, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t4, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t5, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t6, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t7, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t8, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t9, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t11, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t12, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t13, on='user_id', how='left')
user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan, 0)
user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan, 0)
user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype(
    'float') / user1_feature.buy_total.astype('float')
user1_feature[
    'user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype(
        'float') / user1_feature.coupon_received.astype('float')
user1_feature.buy_total = user1_feature.buy_total.replace(np.nan, 0)
user1_feature.coupon_received = user1_feature.coupon_received.replace(
    np.nan, 0)
user1_feature.to_csv('data/user1_feature.csv', index=None)
