import pandas as pd
from datetime import date
import warnings
warnings.filterwarnings('ignore')

dataset1 = pd.read_csv('data/dataset1.csv', keep_default_na=False)
feature1 = pd.read_csv('data/feature1.csv', keep_default_na=False)
dataset2 = pd.read_csv('data/dataset2.csv', keep_default_na=False)
feature2 = pd.read_csv('data/feature2.csv', keep_default_na=False)
dataset3 = pd.read_csv('data/dataset3.csv', keep_default_na=False)
feature3 = pd.read_csv('data/feature3.csv', keep_default_na=False)

############# coupon related feature   #############
"""
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
"""


def calc_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


def get_discount_man(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[0])


def get_discount_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[1])


def is_man_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 0
    else:
        return 1


#dataset3
dataset3['day_of_week'] = dataset3.date_received.astype('str').apply(
    lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
dataset3['day_of_month'] = dataset3.date_received.astype('str').apply(
    lambda x: int(x[6:8]))
dataset3['days_distance'] = dataset3.date_received.astype('str').apply(
    lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days
)
dataset3['discount_man'] = dataset3.discount_rate.apply(get_discount_man)
dataset3['discount_jian'] = dataset3.discount_rate.apply(get_discount_jian)
dataset3['is_man_jian'] = dataset3.discount_rate.apply(is_man_jian)
dataset3['discount_rate'] = dataset3.discount_rate.apply(calc_discount_rate)
d = dataset3[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset3 = pd.merge(dataset3, d, on='coupon_id', how='left')
dataset3.to_csv('data/coupon3_feature.csv', index=None)
#dataset2
dataset2['day_of_week'] = dataset2.date_received.astype('str').apply(
    lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
dataset2['day_of_month'] = dataset2.date_received.astype('str').apply(
    lambda x: int(x[6:8]))
dataset2['days_distance'] = dataset2.date_received.astype('str').apply(
    lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 5, 14)).days
)
dataset2['discount_man'] = dataset2.discount_rate.apply(get_discount_man)
dataset2['discount_jian'] = dataset2.discount_rate.apply(get_discount_jian)
dataset2['is_man_jian'] = dataset2.discount_rate.apply(is_man_jian)
dataset2['discount_rate'] = dataset2.discount_rate.apply(calc_discount_rate)
d = dataset2[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset2 = pd.merge(dataset2, d, on='coupon_id', how='left')
dataset2.to_csv('data/coupon2_feature.csv', index=None)
#dataset1
dataset1['day_of_week'] = dataset1.date_received.astype('str').apply(
    lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
dataset1['day_of_month'] = dataset1.date_received.astype('str').apply(
    lambda x: int(x[6:8]))
dataset1['days_distance'] = dataset1.date_received.astype('str').apply(
    lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 4, 13)).days
)
dataset1['discount_man'] = dataset1.discount_rate.apply(get_discount_man)
dataset1['discount_jian'] = dataset1.discount_rate.apply(get_discount_jian)
dataset1['is_man_jian'] = dataset1.discount_rate.apply(is_man_jian)
dataset1['discount_rate'] = dataset1.discount_rate.apply(calc_discount_rate)
d = dataset1[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset1 = pd.merge(dataset1, d, on='coupon_id', how='left')
dataset1.to_csv('data/coupon1_feature.csv', index=None)
