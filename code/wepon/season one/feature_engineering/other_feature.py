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


############# other feature ##################3
"""
5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""

#for dataset3
t = dataset3[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset3[['user_id', 'coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

t2 = dataset3[['user_id', 'coupon_id', 'date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
t2 = t2[t2.receive_number > 1]
t2['max_date_received'] = t2.date_received.apply(
    lambda s: max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(
    lambda s: min([int(d) for d in s.split(':')]))
t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

t3 = dataset3[['user_id', 'coupon_id', 'date_received']]
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  #those only receive once


t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
    is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
    is_firstlastone)
t3 = t3[[
    'user_id', 'coupon_id', 'date_received',
    'this_month_user_receive_same_coupon_lastone',
    'this_month_user_receive_same_coupon_firstone'
]]

t4 = dataset3[['user_id', 'date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

t5 = dataset3[['user_id', 'coupon_id', 'date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id', 'coupon_id',
                 'date_received']).agg('sum').reset_index()

t6 = dataset3[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t6.rename(columns={'date_received': 'dates'}, inplace=True)


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8])) - date(
                int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


t7 = dataset3[['user_id', 'coupon_id', 'date_received']]
t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[[
    'user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'
]]

other_feature3 = pd.merge(t1, t, on='user_id')
other_feature3 = pd.merge(other_feature3, t3, on=['user_id', 'coupon_id'])
other_feature3 = pd.merge(other_feature3, t4, on=['user_id', 'date_received'])
other_feature3 = pd.merge(
    other_feature3, t5, on=['user_id', 'coupon_id', 'date_received'])
other_feature3 = pd.merge(
    other_feature3, t7, on=['user_id', 'coupon_id', 'date_received'])
other_feature3.to_csv('data/other_feature3.csv', index=None)
print(other_feature3.shape)

#for dataset2
t = dataset2[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset2[['user_id', 'coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

t2 = dataset2[['user_id', 'coupon_id', 'date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
t2 = t2[t2.receive_number > 1]
t2['max_date_received'] = t2.date_received.apply(
    lambda s: max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(
    lambda s: min([int(d) for d in s.split(':')]))
t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

t3 = dataset2[['user_id', 'coupon_id', 'date_received']]
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype(
    'int')
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(
    'int') - t3.min_date_received


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  #those only receive once


t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
    is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
    is_firstlastone)
t3 = t3[[
    'user_id', 'coupon_id', 'date_received',
    'this_month_user_receive_same_coupon_lastone',
    'this_month_user_receive_same_coupon_firstone'
]]

t4 = dataset2[['user_id', 'date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

t5 = dataset2[['user_id', 'coupon_id', 'date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id', 'coupon_id',
                 'date_received']).agg('sum').reset_index()

t6 = dataset2[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t6.rename(columns={'date_received': 'dates'}, inplace=True)


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8])) - date(
                int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


t7 = dataset2[['user_id', 'coupon_id', 'date_received']]
t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[[
    'user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'
]]

other_feature2 = pd.merge(t1, t, on='user_id')
other_feature2 = pd.merge(other_feature2, t3, on=['user_id', 'coupon_id'])
other_feature2 = pd.merge(other_feature2, t4, on=['user_id', 'date_received'])
other_feature2 = pd.merge(
    other_feature2, t5, on=['user_id', 'coupon_id', 'date_received'])
other_feature2 = pd.merge(
    other_feature2, t7, on=['user_id', 'coupon_id', 'date_received'])
other_feature2.to_csv('data/other_feature2.csv', index=None)
print(other_feature2.shape)

#for dataset1
t = dataset1[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset1[['user_id', 'coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

t2 = dataset1[['user_id', 'coupon_id', 'date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
t2 = t2[t2.receive_number > 1]
t2['max_date_received'] = t2.date_received.apply(
    lambda s: max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(
    lambda s: min([int(d) for d in s.split(':')]))
t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

t3 = dataset1[['user_id', 'coupon_id', 'date_received']]
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype(
    'int')
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(
    'int') - t3.min_date_received


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  #those only receive once


t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
    is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
    is_firstlastone)
t3 = t3[[
    'user_id', 'coupon_id', 'date_received',
    'this_month_user_receive_same_coupon_lastone',
    'this_month_user_receive_same_coupon_firstone'
]]

t4 = dataset1[['user_id', 'date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

t5 = dataset1[['user_id', 'coupon_id', 'date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id', 'coupon_id',
                 'date_received']).agg('sum').reset_index()

t6 = dataset1[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(
    ['user_id',
     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t6.rename(columns={'date_received': 'dates'}, inplace=True)


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8])) - date(
                int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(
            int(date_received[0:4]), int(date_received[4:6]),
            int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


t7 = dataset1[['user_id', 'coupon_id', 'date_received']]
t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[[
    'user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'
]]

other_feature1 = pd.merge(t1, t, on='user_id')
other_feature1 = pd.merge(other_feature1, t3, on=['user_id', 'coupon_id'])
other_feature1 = pd.merge(other_feature1, t4, on=['user_id', 'date_received'])
other_feature1 = pd.merge(
    other_feature1, t5, on=['user_id', 'coupon_id', 'date_received'])
other_feature1 = pd.merge(
    other_feature1, t7, on=['user_id', 'coupon_id', 'date_received'])
other_feature1.to_csv('data/other_feature1.csv', index=None)
print(other_feature1.shape)
