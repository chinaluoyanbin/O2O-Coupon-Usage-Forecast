import pandas as pd


def add_weight(p1, p2, w1, w2):
    t1 = pd.read_csv(p1, header=None)
    t2 = pd.read_csv(p2, header=None)
    t1.columns = ['User_id', 'Coupon_id', 'Date_received', 'Proba']
    t2.columns = ['User_id', 'Coupon_id', 'Date_received', 'Proba']

    print('%s: \n%s' % (p1[-24:], t1.head(5)))
    print('%s: \n%s' % (p2[-16:], t2.head(5)))
    
    t1.Proba = t1.Proba * w1 + t2.Proba * w2
    t1.to_csv('C://Users//Administrator//Desktop//o2o-results//weighted_2.csv', index=False, header=False)
    print('weighted: \n%s' % t1.head(5))


if __name__ == '__main__':
    # 预置参数
    TARGET1 = 'C://Users//Administrator//Desktop//o2o-results//xgb_preds_0.79618767.csv'
    TARGET2 = 'C://Users//Administrator//Desktop//o2o-results//gbdt_preds_4.csv'
    WEIGHT1 = 0.65
    WEIGHT2 = 0.35

    # 加权
    add_weight(TARGET1, TARGET2, WEIGHT1, WEIGHT2)
