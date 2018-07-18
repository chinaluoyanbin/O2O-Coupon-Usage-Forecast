import pandas as pd

"""
                    Label(date_received)
        dateset3: 20160701~20160731 (113640),features3 from 20160315~20160630
        dateset2: 20160515~20160615 (258446),features2 from 20160201~20160514  
        dateset1: 20160414~20160514 (138303),features1 from 20160101~20160413
"""

#1754884 record,1053282 with coupon_id,9738 coupon. date_received:20160101~20160615,date:20160101~20160630, 539438 users, 8415 merchants
off_train = pd.read_csv(
    'data/ccf_offline_stage1_train.csv', keep_default_na=False)
off_train.columns = [
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received', 'date'
]
#2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
off_test = pd.read_csv(
    'data/ccf_offline_stage1_test_revised.csv', keep_default_na=False)
off_test.columns = [
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received'
]
#11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
on_train = pd.read_csv(
    'data/ccf_online_stage1_train.csv', keep_default_na=False)
on_train.columns = [
    'user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate',
    'date_received', 'date'
]

dataset3 = off_test
feature3 = off_train[(
    (off_train.date >= '20160315') & (off_train.date <= '20160630')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160315') &
        (off_train.date_received <= '20160630'))]
dataset2 = off_train[(off_train.date_received >= '20160515')
                     & (off_train.date_received <= '20160615')]
feature2 = off_train[
    (off_train.date >= '20160201') & (off_train.date <= '20160514')
    | ((off_train.date == 'null') & (off_train.date_received >= '20160201') &
       (off_train.date_received <= '20160514'))]
dataset1 = off_train[(off_train.date_received >= '20160414')
                     & (off_train.date_received <= '20160514')]
feature1 = off_train[
    (off_train.date >= '20160101') & (off_train.date <= '20160413')
    | ((off_train.date == 'null') & (off_train.date_received >= '20160101') &
       (off_train.date_received <= '20160413'))]

dataset3.to_csv('data/dataset3.csv', index=False)
feature3.to_csv('data/feature3.csv', index=False)
dataset2.to_csv('data/dataset2.csv', index=False)
feature2.to_csv('data/feature2.csv', index=False)
dataset1.to_csv('data/dataset1.csv', index=False)
feature1.to_csv('data/feature1.csv', index=False)
