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


############# merchant related feature   #############
"""
1.merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon

"""

#for dataset3
merchant3 = feature3[[
    'merchant_id', 'coupon_id', 'distance', 'date_received', 'date'
]]

t = merchant3[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant3[merchant3.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][[
    'merchant_id'
]]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant3[merchant3.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][[
    'merchant_id', 'distance'
]]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant3_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t2, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t3, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t5, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t6, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t7, on='merchant_id', how='left')
merchant3_feature = pd.merge(
    merchant3_feature, t8, on='merchant_id', how='left')
merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant3_feature[
    'merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype(
        'float') / merchant3_feature.total_coupon
merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype(
    'float') / merchant3_feature.total_sales
merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant3_feature.to_csv('data/merchant3_feature.csv', index=None)

#for dataset2
merchant2 = feature2[[
    'merchant_id', 'coupon_id', 'distance', 'date_received', 'date'
]]

t = merchant2[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[merchant2.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][[
    'merchant_id'
]]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant2[merchant2.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][[
    'merchant_id', 'distance'
]]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant2_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t2, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t3, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t5, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t6, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t7, on='merchant_id', how='left')
merchant2_feature = pd.merge(
    merchant2_feature, t8, on='merchant_id', how='left')
merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant2_feature[
    'merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype(
        'float') / merchant2_feature.total_coupon
merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype(
    'float') / merchant2_feature.total_sales
merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant2_feature.to_csv('data/merchant2_feature.csv', index=None)

#for dataset1
merchant1 = feature1[[
    'merchant_id', 'coupon_id', 'distance', 'date_received', 'date'
]]

t = merchant1[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[merchant1.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][[
    'merchant_id'
]]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant1[merchant1.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][[
    'merchant_id', 'distance'
]]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant1_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t2, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t3, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t5, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t6, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t7, on='merchant_id', how='left')
merchant1_feature = pd.merge(
    merchant1_feature, t8, on='merchant_id', how='left')
merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant1_feature[
    'merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype(
        'float') / merchant1_feature.total_coupon
merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype(
    'float') / merchant1_feature.total_sales
merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(
    np.nan, 0)  #fillna with 0
merchant1_feature.to_csv('data/merchant1_feature.csv', index=None)
