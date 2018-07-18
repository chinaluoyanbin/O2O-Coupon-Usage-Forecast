import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

dataset1 = pd.read_csv('data/dataset1.csv', keep_default_na=False)
feature1 = pd.read_csv('data/feature1.csv', keep_default_na=False)
dataset2 = pd.read_csv('data/dataset2.csv', keep_default_na=False)
feature2 = pd.read_csv('data/feature2.csv', keep_default_na=False)
dataset3 = pd.read_csv('data/dataset3.csv', keep_default_na=False)
feature3 = pd.read_csv('data/feature3.csv', keep_default_na=False)

##################  user_merchant related feature #########################
"""
4.user_merchant:
      times_user_buy_merchant_before. 
"""
#for dataset3
all_user_merchant = feature3[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature3[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][[
    'user_id', 'merchant_id'
]]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature3[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][[
    'user_id', 'merchant_id'
]]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(
    all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(
    user_merchant3, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(
    user_merchant3, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(
    user_merchant3, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(
    user_merchant3, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(
    np.nan, 0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(
    np.nan, 0)
user_merchant3[
    'user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_received.astype('float')
user_merchant3[
    'user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3[
    'user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype(
        'float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3[
    'user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3.to_csv('data/user_merchant3.csv', index=None)

#for dataset2
all_user_merchant = feature2[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature2[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][[
    'user_id', 'merchant_id'
]]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature2[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature2[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][[
    'user_id', 'merchant_id'
]]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(
    all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(
    user_merchant2, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(
    user_merchant2, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(
    user_merchant2, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(
    user_merchant2, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(
    np.nan, 0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(
    np.nan, 0)
user_merchant2[
    'user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant2.user_merchant_received.astype('float')
user_merchant2[
    'user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2[
    'user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype(
        'float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2[
    'user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype(
        'float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2.to_csv('data/user_merchant2.csv', index=None)

#for dataset2
all_user_merchant = feature1[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][[
    'user_id', 'merchant_id'
]]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][[
    'user_id', 'merchant_id'
]]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(
    all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(
    user_merchant1, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(
    user_merchant1, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(
    user_merchant1, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(
    user_merchant1, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(
    np.nan, 0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(
    np.nan, 0)
user_merchant1[
    'user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1[
    'user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1[
    'user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype(
        'float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1[
    'user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype(
        'float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1.to_csv('data/user_merchant1.csv', index=None)
