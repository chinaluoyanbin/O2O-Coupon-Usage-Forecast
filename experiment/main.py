import pandas as pd
import numpy as np
from datetime import date
import os
import matplotlib.pyplot as plt
import logging

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import PredefinedSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def getDiscount_rate(s):
    if s == 'null':
        return 1
    elif ':' in s:
        temp = s.split(':')
        return 1 - float(temp[1]) / float(temp[0])
    else:
        return float(s)
    
    
def getIsManjian(s):
    if ':' in s:
        return 1
    else:
        return 0


def getDiscountMan(s):
    if s == 'null':
        return np.nan
    elif ':' in s:
        temp = s.split(':')
        return float(temp[0])
    else:
        return np.nan


def getDiscountJian(s):
    if s == 'null':
        return np.nan
    elif ':' in s:
        temp = s.split(':')
        return float(temp[1])
    else:
        return np.nan
    

def get_date_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


def get_label(s):
    s = s.split(':')
    if 'null' in s:
        return 0
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return 0


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  #those only receive once     


def get_preprocessed_data():
    """
    1. 数据预处理
    """
    off_test = pd.read_csv('../input/ccf_offline_stage1_test_revised.csv', keep_default_na=False)
    off_train = pd.read_csv('../input/ccf_offline_stage1_train.csv', keep_default_na=False)
    on_train = pd.read_csv('../input/ccf_online_stage1_train.csv', keep_default_na=False)

    off_test['discount_rate'] = off_test.Discount_rate.apply(getDiscount_rate)
    off_train['discount_rate'] = off_train.Discount_rate.apply(getDiscount_rate)

    print('...preprocess data complete...')

    return off_test, off_train, on_train


def get_split_data():
    """
    2. 划分数据集
        dateset3: 20160701~20160731 (113640), features3 from 20160315~20160630
        dateset2: 20160515~20160615 (258446), features2 from 20160201~20160514  
        dateset1: 20160414~20160514 (138303), features1 from 20160101~20160413
    """
    off_test, off_train, on_train = get_preprocessed_data()
    # for dataset1
    start, end = '20160414', '20160514'
    dataset1 = off_train[(start <= off_train.Date_received) & (off_train.Date_received <= end)]
    # 提取label
    dataset1['label'] = dataset1.Date_received + ':' + dataset1.Date
    dataset1.label = dataset1.label.apply(get_label)
    dataset1.drop(columns=['Date'], inplace=True)
    start, end = '20160101', '20160413'
    off_feature1 = off_train[((start <= off_train.Date) & (off_train.Date <= end)) | ((off_train.Date == 'null') & (start <= off_train.Date_received) & (off_train.Date_received <= end))]
    on_feature1 = on_train[((start <= on_train.Date) & (on_train.Date <= end)) | ((on_train.Date == 'null') & (start <= on_train.Date_received) & (on_train.Date_received <= end))]

    # for dateset2
    start, end = '20160515', '20160615'
    dataset2 = off_train[(start <= off_train.Date_received) & (off_train.Date_received <= end)]
    # 提取label
    dataset2['label'] = dataset2.Date_received + ':' + dataset2.Date
    dataset2.label = dataset2.label.apply(get_label)
    dataset2.drop(columns=['Date'], inplace=True)
    start, end = '20160201', '20160514'
    off_feature2 = off_train[((start <= off_train.Date) & (off_train.Date <= end)) | ((off_train.Date == 'null') & (start <= off_train.Date_received) & (off_train.Date_received <= end))]
    on_feature2 = on_train[((start <= on_train.Date) & (on_train.Date <= end)) | ((on_train.Date == 'null') & (start <= on_train.Date_received) & (on_train.Date_received <= end))]

    # for dataset3
    dataset3 = off_test.copy()
    dataset3.Coupon_id = dataset3.Coupon_id.astype(str)
    dataset3.Date_received = dataset3.Date_received.astype(str)
    start, end = '20160315', '20160630'
    off_feature3 = off_train[((start <= off_train.Date) & (off_train.Date <= end)) | ((off_train.Date == 'null') & (start <= off_train.Date_received) & (off_train.Date_received <= end))]
    on_feature3 = on_train[((start <= on_train.Date) & (on_train.Date <= end)) | ((on_train.Date == 'null') & (start <= on_train.Date_received) & (on_train.Date_received <= end))]

    del off_test, off_train, on_train

    print(dataset1.shape, off_feature1.shape, on_feature1.shape)
    print(dataset2.shape, off_feature2.shape, on_feature2.shape)
    print(dataset3.shape, off_feature3.shape, on_feature3.shape)

    print('...split data complete...')

    return dataset1, off_feature1, on_feature1, dataset2, off_feature2, on_feature2, dataset3, off_feature3, on_feature3


def get_offline_feature(dataset, feature):
    
    """
    3. 提取线下特征
    """

    # weekday
    dataset['weekday'] = dataset.Date_received.astype('str').apply(lambda x: date(int(x[0: 4]), int(x[4: 6]), int(x[6: 8])).weekday() + 1)
    # is_weekend
    dataset['is_weekend'] = dataset.weekday.apply(lambda x: 1 if x in (6, 7) else 0)
    # day
    dataset['day'] = dataset.Date_received.astype('str').apply(lambda x: int(x[6:8]))
    # ========================================================================================================================
    # ===================================== user 字段特征 ====================================================================
    # ========================================================================================================================

    user = feature[['User_id']]
    user.drop_duplicates(inplace=True)

    consume_use_coupon = feature[(feature.Coupon_id != 'null') & (feature.Date != 'null')]  #用户领取优惠券消费信息
    consume_common = feature[(feature.Coupon_id == 'null') & (feature.Date != 'null')]  #用户普通消费信息
    receive_coupon_not_consume = feature[(feature.Coupon_id != 'null') & (feature.Date == 'null')]  #用户领取优惠券但未使用信息
    receive_coupon = feature[feature.Coupon_id != 'null']  #用户领取优惠券信息
    consume = feature[feature.Date != 'null']  #用户消费信息

    # 线下使用优惠券消费的次数
    # u1
    t = consume_use_coupon[['User_id']]
    t['u1'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # # 线下领取优惠券但没有使用的次数
    # # u2
    # t = receive_coupon_not_consume[['User_id']]
    # t['u2'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 使用优惠券次数与没使用优惠券次数比值
    # # u3
    # user.u1.fillna(0, inplace=True)
    # user['u3'] = user.u1 / user.u2

    # 领取优惠券的总次数
    # u4
    t = receive_coupon[['User_id']]
    t['u4'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # 优惠券核销率
    # u5
    user.u1.fillna(0, inplace=True)
    user['u5'] = user.u1 / user.u4

    # # 线下普通消费次数
    # # u6
    # t = consume_common[['User_id']]
    # t['u6'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # user = pd.merge(user, t, how='left', on='User_id')

    # 一共消费多少次
    # u7
    t = consume[['User_id']]
    t['u7'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # # 用户使用优惠券消费占比
    # # u8
    # user['u8'] = user.u1 / user.u7

    # # 线下平均普通消费间隔
    # # u9
    # t = consume_common[['User_id', 'Date']]
    # t = t.groupby('User_id').Date.apply(lambda x: ':'.join(x)).reset_index(name='u9')
    # t['len'] = t.u9.apply(lambda x: len(x.split(':')) - 1)
    # t = t[t.len != 0]
    # t['max_min'] = t.u9.apply(lambda x: max(x.split(':')) + ':' + min(x.split(':')))
    # t['days'] = t.max_min.apply(get_date_gap)
    # t.u9 = t.days.astype(float) / t.len
    # user = pd.merge(user, t[['User_id', 'u9']], how='left', on='User_id')

    # # 线下平均优惠券消费间隔
    # # u10
    # t = consume_use_coupon[['User_id', 'Date']]
    # t = t.groupby('User_id').Date.apply(lambda x: ':'.join(x)).reset_index(name='u10')
    # t['len'] = t.u10.apply(lambda x: len(x.split(':')) - 1)
    # t = t[t.len != 0]
    # t['max_min'] = t.u10.apply(lambda x: max(x.split(':')) + ':' + min(x.split(':')))
    # t['days'] = t.max_min.apply(get_date_gap)
    # t.u10 = t.days.astype(float) / t.len
    # user = pd.merge(user, t[['User_id', 'u10']], how='left', on='User_id')

    # # 15天内平均会普通消费几次
    # # u11
    # user['u11'] = user.u9 / 15
    # user.u11.fillna(0, inplace=True)

    # # 15天内平均会优惠券消费几次
    # # u12
    # user['u12'] = user.u10 / 15
    # user.u12.fillna(0, inplace=True)

    # # 领取优惠券到使用优惠券的平均间隔时间
    # # u13
    # t = consume_use_coupon[['User_id', 'Date_received', 'Date']]
    # t['date_date_received'] = t.Date + ':' + t.Date_received
    # t['u13'] = t.date_date_received.apply(get_date_gap)
    # t = t[['User_id', 'u13']].groupby('User_id').agg('mean').reset_index()
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 在15天内使用掉优惠券的值的大小？
    # # u14
    # user['u14'] = user.u13 / 15

    # # 领取优惠券到使用优惠券间隔小于15天的次数
    # # u15
    # t = consume_use_coupon[['User_id', 'Date_received', 'Date']]
    # t['date_date_received'] = t.Date + ':' + t.Date_received
    # t['u15'] = t.date_date_received.apply(get_date_gap)
    # t = t[t.u15 < 15][['User_id', 'u15']].groupby('User_id').agg('count').reset_index()
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户15天使用掉优惠券的次数除以使用优惠券的次数
    # # u16
    # user['u16'] = user.u1
    # user.u16 = user.u16.replace(0, np.nan)
    # user.u15.fillna(0, inplace=True)
    # user.u16 = user.u15 / user.u16

    # # 用户15天使用掉优惠券的次数除以领取优惠券的总次数
    # # u17
    # user['u17'] = user.u15 / user.u4

    # # 用户15天使用掉优惠券的次数除以领取优惠券未消费的次数
    # # u18
    # user['u18'] = user.u15 / user.u2

    # # 消费优惠券的平均折扣率
    # # u19
    # t = consume_use_coupon[['User_id', 'discount_rate']].groupby('User_id').discount_rate.agg('mean').reset_index(name='u19')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户核销优惠券的最低消费折扣率
    # # u20
    # t = consume_use_coupon[['User_id', 'discount_rate']].groupby('User_id').discount_rate.agg('min').reset_index(name='u20')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户核销优惠券的最高消费折扣率
    # # u21
    # t = consume_use_coupon[['User_id', 'discount_rate']].groupby('User_id').discount_rate.agg('max').reset_index(name='u21')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户领取的优惠券种类
    # # u22
    # t = receive_coupon[['User_id', 'Coupon_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('User_id').Coupon_id.agg('count').reset_index(name='u22')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户平均每种优惠券核销多少张
    # # u23
    # user['u23'] = user.u1 / user.u22

    # # 用户领取优惠券不同商家数量
    # # u24
    # t = receive_coupon[['User_id', 'Merchant_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('User_id').Merchant_id.agg('count').reset_index(name='u24')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户核销优惠券不同商家数量
    # # u25
    # t = consume_use_coupon[['User_id', 'Merchant_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('User_id').Merchant_id.agg('count').reset_index(name='u25')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 用户核销过优惠券的不同商家数量占所有不同商家的比重
    # # u26
    # user.u25.fillna(0, inplace=True)
    # user['u26'] = user.u25 / user.u24

    # # 用户平均核销每个商家多少张优惠券
    # # u27 = u1 / u24
    # user['u27'] = user.u1 / user.u24

    # 领取优惠券到使用优惠券的最小间隔时间
    # u28
    t = consume_use_coupon[['User_id', 'Date_received', 'Date']]
    t['date_date_received'] = t.Date + ':' + t.Date_received
    t['u28'] = t.date_date_received.apply(get_date_gap)
    t = t[['User_id', 'u28']].groupby('User_id').agg('min').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # 领取优惠券到使用优惠券的最大间隔时间
    # u29
    t = consume_use_coupon[['User_id', 'Date_received', 'Date']]
    t['date_date_received'] = t.Date + ':' + t.Date_received
    t['u29'] = t.date_date_received.apply(get_date_gap)
    t = t[['User_id', 'u29']].groupby('User_id').agg('max').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # 领取优惠券到使用优惠券的平均间隔时间
    # u40
    t = consume_use_coupon[['User_id', 'Date_received', 'Date']]
    t['date_date_received'] = t.Date + ':' + t.Date_received
    t['u40'] = t.date_date_received.apply(get_date_gap)
    t = t[['User_id', 'u40']].groupby('User_id').agg('mean').reset_index()
    user = pd.merge(user, t, how='left', on='User_id')

    # 用户消费的不同商家数量
    # u30
    t = consume[['User_id', 'Merchant_id']]
    t.drop_duplicates(inplace=True)
    t = t.groupby('User_id').Merchant_id.agg('count').reset_index(name='u30')
    user = pd.merge(user, t, how='left', on='User_id')

    # 用户优惠券消费的平均距离
    # u31
    t = consume_use_coupon[['User_id', 'Distance']]
    t.replace('null', -1, inplace=True)
    t.Distance = t.Distance.astype('int')
    t.replace(-1, np.nan, inplace=True)
    t = t.groupby('User_id').Distance.agg('mean').reset_index(name='u31')
    user = pd.merge(user, t, how='left', on='User_id')

    # 用户优惠券消费的最小距离
    # u32
    t = consume_use_coupon[['User_id', 'Distance']]
    t.replace('null', -1, inplace=True)
    t.Distance = t.Distance.astype('int')
    t.replace(-1, np.nan, inplace=True)
    t = t.groupby('User_id').Distance.agg('min').reset_index(name='u32')
    user = pd.merge(user, t, how='left', on='User_id')

    # 用户优惠券消费的最大距离
    # u33
    t = consume_use_coupon[['User_id', 'Distance']]
    t.replace('null', -1, inplace=True)
    t.Distance = t.Distance.astype('int')
    t.replace(-1, np.nan, inplace=True)
    t = t.groupby('User_id').Distance.agg('max').reset_index(name='u33')
    user = pd.merge(user, t, how='left', on='User_id')

    # # 满减类型优惠券领取次数
    # # u34
    # t = receive_coupon[receive_coupon.Discount_rate.str.contains(':')][['User_id', 'Discount_rate']]
    # t = t.groupby('User_id').Discount_rate.agg('count').reset_index(name='u34')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 满减类型优惠券消费次数
    # # u35
    # t = consume_use_coupon[consume_use_coupon.Discount_rate.str.contains(':')][['User_id', 'Discount_rate']]
    # t = t.groupby('User_id').Discount_rate.agg('count').reset_index(name='u35')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 满减类型优惠券核销率
    # # u36
    # user.u35.fillna(0, inplace=True)
    # user['u36'] = user.u35 / user.u34

    # # 打折类型优惠券领取次数
    # # u37
    # t = receive_coupon[receive_coupon.Discount_rate.str.contains('\.')][['User_id', 'Discount_rate']]
    # t = t.groupby('User_id').Discount_rate.agg('count').reset_index(name='u37')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 打折类型优惠券消费次数
    # # u38
    # t = consume_use_coupon[consume_use_coupon.Discount_rate.str.contains('\.')][['User_id', 'Discount_rate']]
    # t = t.groupby('User_id').Discount_rate.agg('count').reset_index(name='u38')
    # user = pd.merge(user, t, how='left', on='User_id')

    # # 打折类型优惠券核销率
    # # u39
    # user.u38.fillna(0, inplace=True)
    # user['u39'] = user.u38 / user.u37

    # dataset = pd.merge(dataset, user, how='left', on='User_id')
    # del user

    # # ============================================================================================================================
    # # ================================= user_coupon 双字段特征 ===================================================================
    # # ============================================================================================================================
    # user_coupon = feature[['User_id', 'Coupon_id']]
    # user_coupon.drop_duplicates(inplace=True)

    # # 用户核销过的不同优惠券数量
    # # uc1
    # t = consume_use_coupon[['User_id', 'Coupon_id']]
    # t['uc1'] = 1
    # t = t.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()
    # user_coupon = pd.merge(user_coupon, t, how='left', on=['User_id', 'Coupon_id'])

    # # 用户领取所有不同优惠券数量
    # # uc2
    # t = receive_coupon[['User_id', 'Coupon_id']]
    # t['uc2'] = 1
    # t = t.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()
    # user_coupon = pd.merge(user_coupon, t, how='left', on=['User_id', 'Coupon_id'])

    # # 用户核销过的不同优惠券数量占所有不同优惠券的比重
    # # uc3
    # user_coupon.uc1.fillna(0, inplace=True)
    # user_coupon['uc3'] = user_coupon.uc1 / user_coupon.uc2

    # dataset = pd.merge(dataset, user_coupon, how='left', on=['User_id', 'Coupon_id'])
    # del user_coupon

    # # ==========================================================================================================================
    # # ================================ user_merchant ===========================================================================
    # # ==========================================================================================================================
    user_merchant = feature[['User_id', 'Merchant_id']]
    user_merchant.drop_duplicates(inplace=True)

    # # 核销优惠券用户-商家平均距离
    # # um1
    # t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['User_id', 'Merchant_id', 'Distance']]
    # t.Distance = t.Distance.astype(int)
    # t = t.groupby(['User_id', 'Merchant_id']).Distance.agg('mean').reset_index(name='um1')
    # user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # # 用户核销优惠券中最小用户-商家距离
    # # um2
    # t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['User_id', 'Merchant_id', 'Distance']]
    # t.Distance = t.Distance.astype(int)
    # t = t.groupby(['User_id', 'Merchant_id']).Distance.agg('min').reset_index(name='um2')
    # user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # # 用户核销优惠券中最大用户-商家距离
    # # um3
    # t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['User_id', 'Merchant_id', 'Distance']]
    # t.Distance = t.Distance.astype(int)
    # t = t.groupby(['User_id', 'Merchant_id']).Distance.agg('max').reset_index(name='um3')
    # user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券次数
    # um4
    t = receive_coupon[['User_id', 'Merchant_id']]
    t['um4'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # # 用户领取商家的优惠券后不核销次数
    # # um5
    # t = receive_coupon_not_consume[['User_id', 'Merchant_id']]
    # t['um5'] = 1
    # t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    # user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销次数
    # um6
    t = consume_use_coupon[['User_id', 'Merchant_id']]
    t['um6'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销率
    # um7
    user_merchant.um6.fillna(0, inplace=True)
    user_merchant['um7'] = user_merchant.um6 / user_merchant.um4

    # # 用户对每个商家的不核销次数占用户总的不核销次数的比重
    # # um8
    # t = receive_coupon_not_consume[['User_id']]
    # t['um8'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id'])
    # user_merchant.um5.fillna(0, inplace=True)
    # user_merchant.um8 = user_merchant.um5 / user_merchant.um8

    # 用户在商店总共消费过几次
    # um9
    t = consume[['User_id', 'Merchant_id']]
    t['um9'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # 用户在商店普通消费次数
    # um10
    t = consume_common[['User_id', 'Merchant_id']]
    t['um10'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])

    # # 用户当天在此商店领取的优惠券数目
    # # um11
    # t = dataset[dataset.Date_received != 'null'][['User_id', 'Merchant_id', 'Date_received']]
    # t['um11'] = 1
    # t = t.groupby(['User_id', 'Merchant_id', 'Date_received']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id', 'Merchant_id', 'Date_received'])

    # 用户商家数量统计
    # um12
    t = feature[['User_id', 'Merchant_id']]
    t['um12'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    user_merchant = pd.merge(user_merchant, t, how='left', on=['User_id', 'Merchant_id'])    

    # 用户商家优惠券消费占总消费次数的比重
    # um13
    user_merchant['um13'] = user_merchant.um6 / user_merchant.um9

    # # 用户商家消费次数占用户商家总数量的比重
    # # um14
    # user_merchant.um9.fillna(0, inplace=True)
    # user_merchant['um14'] = user_merchant.um9 / user_merchant.um12

    # # 用户商家优惠券消费次数占用户商家数量的比重
    # # um15
    # user_merchant['um15'] = user_merchant.um6 / user_merchant.um12

    # dataset = pd.merge(dataset, user_merchant, how='left', on=['User_id', 'Merchant_id'])
    # del user_merchant

    # # =========================================================================================================================
    # # ================================ user_discount_rate =====================================================================
    # # =========================================================================================================================
    # user_discount = feature[['User_id', 'Discount_rate']]
    # user_discount.drop_duplicates(inplace=True)

    # # 不同打折优惠券领取次数
    # # ud1
    # t = receive_coupon[['User_id', 'Discount_rate']]
    # t['ud1'] = 1
    # t = t.groupby(['User_id', 'Discount_rate']).agg('sum').reset_index()
    # user_discount = pd.merge(user_discount, t, how='left', on=['User_id', 'Discount_rate'])

    # # 不同打折优惠券使用次数
    # # ud2
    # t = consume_use_coupon[['User_id', 'Discount_rate']]
    # t['ud2'] = 1
    # t = t.groupby(['User_id', 'Discount_rate']).agg('sum').reset_index()
    # user_discount = pd.merge(user_discount, t, how='left', on=['User_id', 'Discount_rate'])

    # # 不同打折优惠券不使用次数
    # # ud3
    # t = receive_coupon_not_consume[['User_id', 'Discount_rate']]
    # t['ud3'] = 1
    # t = t.groupby(['User_id', 'Discount_rate']).agg('sum').reset_index()
    # user_discount = pd.merge(user_discount, t, how='left', on=['User_id', 'Discount_rate'])

    # # 不同打折优惠券使用率
    # # ud4
    # user_discount.ud2.fillna(0, inplace=True)
    # user_discount['ud4'] = user_discount.ud2 / user_discount.ud1

    # dataset = pd.merge(dataset, user_discount, how='left', on=['User_id', 'Discount_rate'])
    # del user_discount

    # # ===========================================================================================================================
    # # ================================ merchant 字段特征 ========================================================================
    # # ===========================================================================================================================
    merchant = feature[['Merchant_id']]
    merchant.drop_duplicates(inplace=True)

    # 商家被消费次数
    # m1
    t = consume[['Merchant_id']]
    t['m1'] = 1
    t = t.groupby('Merchant_id').agg('sum').reset_index()
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商家被优惠券消费次数
    # m2
    t = consume_use_coupon[['Merchant_id']]
    t['m2'] = 1
    t = t.groupby('Merchant_id').agg('sum').reset_index()
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')


    # # 商户被普通消费次数
    # # m3
    # t = consume_common[['Merchant_id']]
    # t['m3'] = 1
    # t = t.groupby('Merchant_id').agg('sum').reset_index()
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商户优惠券被领取次数
    # m4
    t = receive_coupon[['Merchant_id']]
    t['m4'] = 1
    t = t.groupby('Merchant_id').agg('sum').reset_index()
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商家优惠券被领取后的核销率
    # m5 = m2 / m4
    merchant.m2.fillna(0, inplace=True)
    merchant['m5'] = merchant.m2 / merchant.m4

    # # 商家优惠券被领取后不核销次数
    # # m6
    # t = receive_coupon_not_consume[['Merchant_id']]
    # t['m6'] = 1
    # t = t.groupby('Merchant_id').agg('sum').reset_index()
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商户当天优惠券被领取次数
    # # m7
    # t = dataset[dataset.Date_received != 'null'][['Merchant_id', 'Date_received']]
    # t['m7'] = 1
    # t = t.groupby(['Merchant_id', 'Date_received']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id', 'Date_received'])

    # # 商户当天优惠券领取人数
    # # m8
    # t = dataset[dataset.Date_received != 'null'][['Merchant_id', 'User_id', 'Date_received']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby(['Merchant_id', 'Date_received']).User_id.agg('count').reset_index(name='m8')
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id', 'Date_received'])

    # # 商家优惠券核销的平均消费折扣率
    # # m9
    # t = consume_use_coupon[['Merchant_id', 'discount_rate']]
    # t = t.groupby('Merchant_id').discount_rate.agg('mean').reset_index(name='m9')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家优惠券核销的最小消费折扣率
    # # m10
    # t = consume_use_coupon[['Merchant_id', 'discount_rate']]
    # t = t.groupby('Merchant_id').discount_rate.agg('min').reset_index(name='m10')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家优惠券核销的最大消费折扣率
    # # m11
    # t = consume_use_coupon[['Merchant_id', 'discount_rate']]
    # t = t.groupby('Merchant_id').discount_rate.agg('max').reset_index(name='m11')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家优惠券核销的不同的用户数量
    # # m12
    # t = consume_use_coupon[['Merchant_id', 'User_id']]
    # t = t.groupby('Merchant_id').User_id.agg('count').reset_index(name='m12')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家优惠券领取的不同的用户数量
    # # m13
    # t = receive_coupon[['Merchant_id', 'User_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('Merchant_id').User_id.agg('count').reset_index(name='m13')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 核销商家优惠券的不同用户数量其占领取不同的用户比重
    # # m14 = m12 / m13
    # merchant.m12.fillna(0, inplace=True)
    # merchant['m14'] = merchant.m12 / merchant.m13

    # # 商家优惠券平均每个用户核销多少张
    # # m15 = m2 / m13
    # merchant['m15'] = merchant.m2 / merchant.m13

    # # 商家被核销过的不同优惠券数量
    # # m16
    # t = consume_use_coupon[['Merchant_id', 'Coupon_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('Merchant_id').Coupon_id.agg('count').reset_index(name='m16')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家领取过的不同优惠券数量
    # # m17
    # t = receive_coupon[['Merchant_id', 'Coupon_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby('Merchant_id').Coupon_id.agg('count').reset_index(name='m17')
    # merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    # # m18 = m16 / m17
    # merchant.m16.fillna(0, inplace=True)
    # merchant['m18'] = merchant.m16 / merchant.m17

    # # 商家被核销优惠券的平均时间
    # # m19
    # t = consume_use_coupon[['Merchant_id', 'Date']]
    # t = t.groupby('Merchant_id').Date.apply(lambda x: ':'.join(x)).reset_index(name='m19')
    # t['len'] = t.m19.apply(lambda x: len(x.split(':')) - 1)
    # t = t[t.len != 0]
    # t['max_min'] = t.m19.apply(lambda x: max(x.split(':')) + ':' + min(x.split(':')))
    # t['days'] = t.max_min.apply(get_date_gap)
    # t.m19 = t.days.astype(float) / t.len
    # merchant = pd.merge(merchant, t[['Merchant_id', 'm19']], how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家平均距离
    # m20
    t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['Merchant_id', 'Distance']]
    t.Distance = t.Distance.astype(int)
    t = t.groupby('Merchant_id').Distance.agg('mean').reset_index(name='m20')
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最小距离
    # m21
    t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['Merchant_id', 'Distance']]
    t.Distance = t.Distance.astype(int)
    t = t.groupby('Merchant_id').Distance.agg('min').reset_index(name='m21')
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最大距离
    # m22
    t = consume_use_coupon[consume_use_coupon.Distance != 'null'][['Merchant_id', 'Distance']]
    t.Distance = t.Distance.astype(int)
    t = t.groupby('Merchant_id').Distance.agg('max').reset_index(name='m22')
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    # 商家被优惠券消费次数占消费次数比重
    # m23
    merchant['m23'] = merchant.m2 / merchant.m1

    # 商家被消费的不同的用户数量
    # m24
    t = consume[['Merchant_id', 'User_id']]
    t = t.groupby('Merchant_id').User_id.agg('count').reset_index(name='m24')
    merchant = pd.merge(merchant, t, how='left', on='Merchant_id')

    dataset = pd.merge(dataset, merchant, how='left', on='Merchant_id')
    del merchant

    # # ==========================================================================================================================
    # # ================================ coupon 字段特征 =========================================================================
    # # ==========================================================================================================================
    coupon = receive_coupon[['Coupon_id']]
    coupon.drop_duplicates(inplace=True)

    # # 此优惠券一共发行多少张
    # # c1
    # t = receive_coupon[['Coupon_id']]
    # t['c1'] = 1
    # t = t.groupby('Coupon_id').agg('sum').reset_index()
    # coupon = pd.merge(coupon, t, how='left', on='Coupon_id')

    # # 此优惠券一共被使用多少张
    # # c2
    # t = consume_use_coupon[['Coupon_id']]
    # t['c2'] = 1
    # t = t.groupby('Coupon_id').agg('sum').reset_index()
    # coupon = pd.merge(coupon, t, how='left', on='Coupon_id')

    # # 优惠券使用率
    # # c3
    # coupon.c2.fillna(0, inplace=True)
    # coupon['c3'] = coupon.c2 / coupon.c1

    # # 没有使用的数目
    # # c4
    # t = receive_coupon_not_consume[['Coupon_id']]
    # t['c4'] = 1
    # t = t.groupby('Coupon_id').agg('sum').reset_index()
    # coupon = pd.merge(coupon, t, how='left', on='Coupon_id')

    # # 此优惠券在当天发行了多少张
    # # c5
    # t = dataset[dataset.Coupon_id != 'null'][['Coupon_id', 'Date_received']]
    # t['c5'] = 1
    # t = t.groupby(['Coupon_id', 'Date_received']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['Coupon_id', 'Date_received'])

    # 优惠券类型(直接优惠为0, 满减为1)
    # c6
    dataset['c6'] = dataset.Discount_rate.apply(getIsManjian)

    # # 不同打折优惠券领取次数
    # # c7
    # t = receive_coupon[['Discount_rate']]
    # t['c7'] = 1
    # t = t.groupby('Discount_rate').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on='Discount_rate')

    # # 不同打折优惠券使用次数
    # # c8
    # t = consume_use_coupon[['Discount_rate']]
    # t['c8'] = 1
    # t = t.groupby('Discount_rate').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on='Discount_rate')

    # # 不同打折优惠券不使用次数
    # # c9
    # t = receive_coupon_not_consume[['Discount_rate']]
    # t['c9'] = 1
    # t = t.groupby('Discount_rate').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on='Discount_rate')

    # # 不同打折优惠券使用率
    # # c10 = c8 / c7
    # dataset.c8.fillna(0, inplace=True)
    # dataset['c10'] = dataset.c8 / dataset.c7

    # # 优惠券核销平均时间
    # # c11
    # t = consume_use_coupon[['Coupon_id', 'Date']]
    # t = t.groupby('Coupon_id').Date.apply(lambda x: ':'.join(x)).reset_index(name='c11')
    # t['len'] = t.c11.apply(lambda x: len(x.split(':')) - 1)
    # t = t[t.len != 0]
    # t['max_min'] = t.c11.apply(lambda x: max(x.split(':')) + ':' + min(x.split(':')))
    # t['days'] = t.max_min.apply(get_date_gap)
    # t.c11 = t.days.astype(float) / t.len
    # coupon = pd.merge(coupon, t[['Coupon_id', 'c11']], how='left', on='Coupon_id')

    # 满减优惠券中的满
    # c12
    dataset['c12'] = dataset.Discount_rate.apply(getDiscountMan)

    # 满减优惠券中的减
    # c13
    dataset['c13'] = dataset.Discount_rate.apply(getDiscountJian)

    dataset = pd.merge(dataset, coupon, how='left', on='Coupon_id')
    del coupon

    # # ==========================================================================================================================
    # # ================================ other feature ===========================================================================
    # # ==========================================================================================================================
    # 用户领取的所有优惠券数目(label窗)
    # o1
    t = dataset[['User_id']]
    t['o1'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on='User_id')

    # 用户领取的特定优惠券数目
    # o2
    t = dataset[['User_id', 'Coupon_id']]
    t['o2'] = 1
    t = t.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on=['User_id', 'Coupon_id'])

    # # 用户下一次领取的平均时间间隔
    # # o3
    # t = dataset[['User_id', 'Date_received']]
    # t = t.groupby('User_id').Date_received.apply(lambda x: ':'.join(x)).reset_index(name='o3')
    # t['len'] = t.o3.apply(lambda x: len(x.split(':')) - 1)
    # t = t[t.len != 0]
    # t['max_min'] = t.o3.apply(lambda x: max(x.split(':')) + ':' + min(x.split(':')))
    # t['days'] = t.max_min.apply(get_date_gap)
    # t.o3 = t.days.astype(float) / t.len
    # dataset = pd.merge(dataset, t[['User_id', 'o3']], how='left', on=['User_id'])

    # # 用户领取特定商家的优惠券数目
    # # o4
    # t = dataset[['User_id', 'Merchant_id']]
    # t['o4'] = 1
    # t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id', 'Merchant_id'])

    # # 用户领取的不同商家数目
    # # o5
    # t = dataset[['User_id', 'Merchant_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby(['User_id']).Merchant_id.agg('count').reset_index(name='o5')
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # 用户当天领取的优惠券数目
    # o6
    t = dataset[['User_id', 'Date_received']]
    t['o6'] = 1
    t = t.groupby(['User_id', 'Date_received']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on=['User_id', 'Date_received'])

    # 用户当天领取的特定优惠券数目
    # o7
    t = dataset[['User_id', 'Coupon_id', 'Date_received']]
    t['o7'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on=['User_id', 'Coupon_id', 'Date_received'])

    # # 用户领取的所有优惠券种类数目
    # # o8
    # t = dataset[['User_id', 'Coupon_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby(['User_id']).Coupon_id.agg('count').reset_index(name='o8')
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 商家被领取的优惠券数目
    # # o9
    # t = dataset[['Merchant_id']]
    # t['o9'] = 1
    # t = t.groupby(['Merchant_id']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id'])

    # # 商家被领取的特定优惠券数目
    # # o10
    # t = dataset[['Merchant_id', 'Coupon_id']]
    # t['o10'] = 1
    # t = t.groupby(['Merchant_id', 'Coupon_id']).agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id', 'Coupon_id'])

    # # 商家被多少不同用户领取的数目
    # # o11
    # t = dataset[['Merchant_id', 'User_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby(['Merchant_id']).User_id.agg('count').reset_index(name='o11')
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id'])

    # # 商家发行的所有优惠券种类数目
    # # o12
    # t = dataset[['Merchant_id', 'Coupon_id']]
    # t.drop_duplicates(inplace=True)
    # t = t.groupby(['Merchant_id']).Coupon_id.agg('count').reset_index(name='o12')
    # dataset = pd.merge(dataset, t, how='left', on=['Merchant_id'])

    # 是否是当月领取相同优惠券中的第一张、最后一张
    # o13, o14
    t = dataset[['User_id', 'Coupon_id', 'Date_received']]
    t.Date_received = t.Date_received.astype('str')
    t = t.groupby(['User_id', 'Coupon_id']).Date_received.agg(lambda x: ':'.join(x)).reset_index()
    t['receive_number'] = t.Date_received.apply(lambda s: len(s.split(':')))
    t = t[t.receive_number > 1]
    t['max_date_received'] = t.Date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t['min_date_received'] = t.Date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t = t[['User_id', 'Coupon_id', 'max_date_received', 'min_date_received']]

    t1 = dataset[['User_id', 'Coupon_id', 'Date_received']]
    t1 = pd.merge(t1, t, how='left', on=['User_id', 'Coupon_id'])
    t1['o13'] = t1.max_date_received.astype('float') - t1.Date_received.astype('float')
    t1['o14'] = t1.Date_received.astype('float') - t1.min_date_received.astype('float')

    t1.o13 = t1.o13.apply(is_firstlastone)
    t1.o14 = t1.o14.apply(is_firstlastone)
    t1 = t1[['User_id', 'Coupon_id', 'Date_received', 'o13', 'o14']]
    dataset = pd.merge(dataset, t1, how='left', on=['User_id', 'Coupon_id', 'Date_received'])

    # # 最近一次消费到当前领券的时间间隔
    # # o15
    # t = dataset[['User_id', 'Date_received']]
    # t.drop_duplicates(inplace=True)
    # t1 = consume[['User_id', 'Date']]
    # t1 = t1.groupby('User_id').Date.agg('max').reset_index(name='last_consume')
    # t = pd.merge(t, t1, how='left', on='User_id')
    # t.last_consume.fillna('null', inplace=True)
    # t = t[t.last_consume != 'null']
    # t['o15'] = t.Date_received.astype('str') + ':' + t.last_consume.astype('str')
    # t.o15 = t.o15.apply(get_date_gap)
    # dataset = pd.merge(dataset, t[['User_id', 'Date_received', 'o15']], how='left', on=['User_id', 'Date_received'])

    # # # 最近一次优惠券消费到当前领券的时间间隔
    # # # o16
    # t = dataset[['User_id', 'Date_received']]
    # t.drop_duplicates(inplace=True)
    # t1 = consume_use_coupon[['User_id', 'Date']]
    # t1 = t1.groupby('User_id').Date.agg('max').reset_index(name='last_consume')
    # t = pd.merge(t, t1, how='left', on='User_id')
    # t.last_consume.fillna('null', inplace=True)
    # t = t[t.last_consume != 'null']
    # t['o16'] = t.Date_received.astype('str') + ':' + t.last_consume.astype('str')
    # t.o16 = t.o16.apply(get_date_gap)
    # dataset = pd.merge(dataset, t[['User_id', 'Date_received', 'o16']], how='left', on=['User_id', 'Date_received'])

    del t, t1, consume_use_coupon, consume_common, receive_coupon_not_consume, receive_coupon, consume
    print('...get offline feature complete...')
    return dataset


def get_online_feature(dataset, feature):

    """
    4. 提取线上特征
    """

    # # 用户线上操作次数
    # # on_u1
    # t = feature[['User_id']]
    # t['on_u1'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 用户线上点击次数
    # # on_u2
    # t = feature[feature.Action == 0][['User_id']]
    # t['on_u2'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 用户线上点击率
    # # on_u3
    # dataset.on_u2.fillna(0, inplace=True)
    # dataset['on_u3'] = dataset.on_u2 / dataset.on_u1

    # 用户线上购买次数
    # on_u4
    t = feature[feature.Action == 1][['User_id']]
    t['on_u4'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 用户线上购买率
    # # on_u5
    # dataset.on_u4.fillna(0, inplace=True)
    # dataset['on_u5'] = dataset.on_u4 / dataset.on_u1

    # # 用户线上领取次数
    # # on_u6
    # t = feature[feature.Action == 2][['User_id']]
    # t['on_u6'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 用户线上领取率
    # # on_u7
    # dataset.on_u6.fillna(0, inplace=True)
    # dataset['on_u7'] = dataset.on_u6 / dataset.on_u1

    # # 用户线上领取优惠券不消费次数
    # # on_u8
    # t = feature[(feature.Date == 'null') & (feature.Action == 2)][['User_id']]
    # t['on_u8'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # 用户线上优惠券核销次数
    # on_u9
    t = feature[(feature.Date != 'null') & (feature.Action == 2)][['User_id']]
    t['on_u9'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t, how='left', on=['User_id'])

    # # 用户线上优惠券核销率
    # # on_u10
    # dataset.on_u9.fillna(0, inplace=True)
    # dataset['on_u10'] = dataset.on_u9 / dataset.on_u6

    # # 用户线下不消费次数占线上线下总的不消费次数的比重
    # # on_u11
    # dataset.u2.fillna(0, inplace=True)
    # dataset['on_u11'] = dataset.u2 / (dataset.u2 + dataset.on_u8)

    # # 用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重
    # # on_u12
    # dataset.u1.fillna(0, inplace=True)
    # dataset['on_u12'] = dataset.u1 / (dataset.u1 + dataset.on_u9)

    # # 用户线下领取的记录数量占总的记录数量的比重
    # # on_u13
    # dataset.u4.fillna(0, inplace=True)
    # dataset['on_u13'] = dataset.u4 / (dataset.u4 + dataset.on_u6)

    # # 用户线上普通消费次数
    # # on_u14
    # t = feature[(feature.Action == 1) & (feature.Coupon_id == 'null')][['User_id']]
    # t['on_u14'] = 1
    # t = t.groupby('User_id').agg('sum').reset_index()
    # dataset = pd.merge(dataset, t, how='left', on='User_id')
    
    del t
    print('...get online feature complete...')
    
    return dataset


def get_dataset():
    """
    5. 生成数据集
    """
    dataset1, off_feature1, on_feature1, dataset2, off_feature2, on_feature2, dataset3, off_feature3, on_feature3 = get_split_data()

    dataset1 = get_offline_feature(dataset1, off_feature1)
    dataset1 = get_online_feature(dataset1, on_feature1)

    dataset2 = get_offline_feature(dataset2, off_feature2)
    dataset2 = get_online_feature(dataset2, on_feature2)

    dataset3 = get_offline_feature(dataset3, off_feature3)
    dataset3 = get_online_feature(dataset3, on_feature3)
    
    # dataset3去重
    dataset3.drop_duplicates(inplace=True)

    Submission = dataset3[['User_id', 'Coupon_id', 'Date_received']]

    # print(dataset1.shape, dataset2.shape, dataset3.shape)
    # (137167, 112) (258446, 112) (113640, 112)
    # (139785, 129) (262240, 129) (116204, 128)

    del off_feature1, off_feature2, off_feature3
    del on_feature1, on_feature2, on_feature3

    # one-hot处理
    weekday_dummies = pd.get_dummies(dataset1.weekday)
    weekday_dummies.columns = ['weekday_' + str(i) for i in range(1, 8)]
    dataset1 = pd.concat([dataset1, weekday_dummies], axis=1)
    weekday_dummies = pd.get_dummies(dataset2.weekday)
    weekday_dummies.columns = ['weekday_' + str(i) for i in range(1, 8)]
    dataset2 = pd.concat([dataset2, weekday_dummies], axis=1)
    weekday_dummies = pd.get_dummies(dataset3.weekday)
    weekday_dummies.columns = ['weekday_' + str(i) for i in range(1, 8)]
    dataset3 = pd.concat([dataset3, weekday_dummies], axis=1)

    # 删掉没用的列
    drop_columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'weekday']
    dataset1.drop(columns=drop_columns, inplace=True)
    dataset1.Distance.replace('null', -999, inplace=True)
    dataset1.Distance = dataset1.Distance.astype(int)
    dataset1.Distance.replace(-999, np.nan, inplace=True)
    dataset2.drop(columns=drop_columns, inplace=True)
    dataset2.Distance.replace('null', -999, inplace=True)
    dataset2.Distance = dataset2.Distance.astype(int)
    dataset2.Distance.replace(-999, np.nan, inplace=True)
    dataset3.drop(columns=drop_columns, inplace=True)
    dataset3.Distance.replace('null', -999, inplace=True)
    dataset3.Distance = dataset3.Distance.astype(int)
    dataset3.Distance.replace(-999, np.nan, inplace=True)

    print('...get dataset complete...')

    return dataset1, dataset2, dataset3, Submission


def read_dataset():
    dataset1_path = '../dataset/dataset1.csv'
    dataset2_path = '../dataset/dataset2.csv'
    dataset3_path = '../dataset/dataset3.csv'
    Submission_path = '../dataset/Submission.csv'

    if os.path.exists(dataset1_path) and os.path.exists(dataset2_path) and os.path.exists(dataset3_path):
        dataset1 = pd.read_csv(dataset1_path)
        dataset2 = pd.read_csv(dataset2_path)
        dataset3 = pd.read_csv(dataset3_path)
        Submission = pd.read_csv(Submission_path)

        print(dataset1.shape, dataset2.shape, dataset3.shape, Submission.shape)        
        print('...read dataset complete...')

        return dataset1, dataset2, dataset3, Submission
    else:
        dataset1, dataset2, dataset3, Submission = get_dataset()

        if os.path.exists('../dataset'):
            dataset1.to_csv(dataset1_path, index=False)
            dataset2.to_csv(dataset2_path, index=False)
            dataset3.to_csv(dataset3_path, index=False)
            Submission.to_csv(Submission_path, index=False)

        print(dataset1.shape, dataset2.shape, dataset3.shape, Submission.shape)
        print('...read dataset complete...')

        return dataset1, dataset2, dataset3, Submission


def analyze_dataset():
    logging.info('Feature Analyze')
    dataset1, dataset2, dataset3, Submission = read_dataset()
    dataset = pd.concat([dataset1, dataset2], axis=0)
    # 分析训练集
    datasetHeight = dataset.shape[0]
    datasetWidth = dataset.shape[1]
    logging.info('Dataset Missing Value Statistics')
    logging.info('Height')
    logging.info('%14s%14s%14s' % ('feature', 'quantity', 'percentage'))
    deletedFeatures = []
    heightStats = dataset.count(axis=0)
    for el in heightStats.index:
        percent = heightStats[el] / datasetHeight
        if percent <= 0.1:
            deletedFeatures.append(el)
        logging.info('%14s%14d%14.3f' % (el, heightStats[el], percent))
    logging.info('deletedFeatures: %s' % deletedFeatures)

    logging.info('Width')
    deletedRows = []
    widthStats = dataset.count(axis=1)
    for i in widthStats.index:
        percent = widthStats.iloc[i] / datasetWidth
        if percent <= 0.1:
            deletedRows.append(i)
    logging.info('deletedRows: %s' % deletedRows)
    logging.info('deletedRows length: %d' % len(deletedRows))


def get_model_input(model='gbdt', set_param=False, train=False, pred=False):
    dataset1, dataset2, dataset3, Submission = read_dataset()

    if model != 'xgb':
        # 如果模型不是xgb填充-999
        dataset1.fillna(-999, inplace=True)
        dataset2.fillna(-999, inplace=True)
        dataset3.fillna(-999, inplace=True)
    dataset = pd.concat([dataset1, dataset2], axis=0)
    if set_param:
        Y_train = dataset1[['label']]
        X_train = dataset1.drop(columns='label')
        return X_train, Y_train
    if train:
        split_point = len(dataset) * 4 // 5
        train_data = dataset.iloc[: split_point, :]
        test_data = dataset.iloc[split_point:, :]
        Y_train = train_data[['label']]
        X_train = train_data.drop(columns='label')
        Y_test = test_data[['label']]
        X_test = test_data.drop(columns='label')
        print(dataset.shape, train_data.shape, test_data.shape)
        return X_train, Y_train, X_test, Y_test
    if pred:
        Y_train = dataset[['label']]
        X_train = dataset.drop(columns='label')
        X_pred = dataset3
        return X_train, Y_train, X_pred, Submission


def get_model_parameter(model='rf', search=False):
    if model == 'rf':
        estimator = RandomForestClassifier(
            random_state=0,
            verbose=32,
            n_estimators=350,
            criterion='gini',
            max_features=19,
            min_samples_leaf=31,
            min_samples_split=2,
            max_depth=30
        )
        param_grid = {
            # 'n_estimators': [i for i in range(70, 571, 100)]
            # 'criterion': ['gini', 'entropy']
            # 'max_depth': [i for i in range(20, 40, 5)]
            # 'min_samples_split': [i for i in range(2, 53, 10)]
            # 'min_samples_leaf': [i for i in range(1, 52, 5)]
            # 'max_features': [i for i in range(5, 13)]
            # 'max_features': [i for i in range(19, 30)]
        }
    elif model == 'gbdt':
        estimator = GradientBoostingClassifier(
            random_state=621,
            verbose=1,
            learning_rate=0.05,
            n_estimators=240,
            subsample=0.7,
            max_features=11,
            min_samples_leaf=50,
            min_samples_split=190,
            max_depth=9
        )
        param_grid = {
            # 'n_estimators': [i for i in range(60, 81, 10)]
            # 'subsample': [.7, .75, .8, .85, .9]
            # 'max_features': [i for i in range(9, 14)]
            # 'min_samples_leaf': [i for i in range(1, 91, 10)]
            # 'min_samples_leaf': [i for i in range(81, 162, 10)]
            # 'min_samples_split': [i for i in range(22, 103, 10)]
            # 'max_depth': [i for i in range(5, 11)]
            # 'max_depth': [9, 10, 11]
            # 'min_samples_split': [185, 190, 195],
            # 'min_samples_leaf': [45, 50, 55]
            # 'max_feature': [10, 11]
        }
    elif model == 'xgb':
        estimator = XGBClassifier(
            random_state=0,
            booster='gbtree',
            objective='rank:pairwise',
            tree_method='exact',
            learning_rate=0.3,
            n_estimators=100,
            gamma=0.2
        )
        param_grid = {
            # 'n_estimators': [i for i in range(80, 101, 10)]
            # 'gamma': [.05, .1, .15, .2, .3, .4]
            'min_child_weight': [i for i in range(1, 102, 10)]
        }
    # Grid Search
    if search:
        X_train, Y_train = get_model_input(model, set_param=True)
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=48,
            cv=5)
        grid.fit(X_train, Y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        logging.info('Model: %s' % model)
        logging.info('means: %s' % means)
        logging.info('stds: %s' % stds)
        logging.info('scores: %s' % grid.grid_scores_)
        logging.info('best params: %s' % grid.best_params_)
        logging.info('best scores: %s' % grid.  )
        logging.info('--------------------------------------------------------')

        plt.figure(figsize=(16, 4))
        if len(param_grid.keys()) == 1:
            xdata = list(param_grid.values())[0]
        else:
            xdata = [i for i in range(1, len(means) + 1)]
        plt.subplot(1, 2, 1)
        plt.title('Mean AUC on Validation')
        plt.plot(xdata, means)
        plt.ylabel('Mean AUC')
        plt.subplot(1, 2, 2)
        plt.title('Std AUC on Validation')
        plt.plot(xdata, stds)
        plt.ylabel('Std AUC')
        plt.savefig(model + '_' + list(param_grid.keys())[0] + '.png')
    return estimator


def training(model='rf'):
    X_train, Y_train, X_test, Y_test = get_model_input(model, train=True)
    if model == 'rf':
        estimator = get_model_parameter(model='rf')
    elif model == 'gbdt':
        estimator = get_model_parameter(model='gbdt')
    estimator.fit(X_train, Y_train)
    Y_pred = estimator.predict(X_test)
    Y_pred_prob = estimator.predict_proba(X_test)[:, 1]

    logging.info('Estimator: %s' % model)
    logging.info('Params: %s' % estimator.get_params())
    logging.info('Accuracy: %f' % metrics.accuracy_score(Y_test, Y_pred))
    logging.info('AUC: %f' % metrics.roc_auc_score(Y_test, Y_pred_prob))


def prediction(model='rf'):
    X_train, Y_train, X_pred, Submission = get_model_input(model=model, pred=True)
    if model == 'rf':
        estimator = get_model_parameter(model='rf')
    elif model == 'gbdt':
        estimator = get_model_parameter(model='gbdt')
    estimator.fit(X_train, Y_train)
    Y_pred_prob = estimator.predict_proba(X_pred)[:, 1]
    Submission['Proba'] = Y_pred_prob
    Submission.to_csv(model + '_preds.csv', index=False, header=False)

    featureImportance = pd.Series(estimator.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    featureImportance.to_csv(model + '_featureImportance.csv')
    plt.figure()
    featureImportance.plot(kind='bar', title='Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Score')
    plt.savefig(model + '_featureImportance.png')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_output = logging.FileHandler('main.log', mode='a')
    logger_output.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s]: %(message)s")
    logger_output.setFormatter(formatter)
    logger.addHandler(logger_output)

    # dataset1, dataset2, dataset3, Submission = read_dataset()
    # print(dataset1.shape, dataset2.shape, dataset3.shape)
    # dataset1.drop_duplicates(inplace=True)
    # dataset2.drop_duplicates(inplace=True)
    # dataset3.drop_duplicates(inplace=True)
    # print(dataset1.shape, dataset2.shape, dataset3.shape)
    # analyze_dataset()

    # estimator = get_model_parameter(model='gbdt', search=True)
    training(model='gbdt')
    # prediction(model='gbdt')
