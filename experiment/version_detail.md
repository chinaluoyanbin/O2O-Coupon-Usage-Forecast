﻿# result

## xgbst_preds

| versions | details                                              | Score      |
| -------- | ---------------------------------------------------- | ---------- |
| _1       | wepon特征工程中的基本特征                            | 0.78650239 |
| _2       | _1 基础上，添加 o10-商家被领取的特定优惠券数目       | 0.78424560 |
| _3       | _2 基础上，添加线上特征 on_u1~on_u10, on_u12, on_u13 | 0.78329692 |
| _4       | _3 基础上，添加了其他的 other_feature                | 0.79415653 |
| _5       | _4 基础上，自己调整了参数                            | 0.79102171 |
| _6       | _4 基础上，将n_estimators 从 2264 调整到 3500        | 0.79211301 |
| _7       | wepon版本的xgb                                       | 0.79618767 |

## gbdt_preds

| versions | details                                                                                 | Score      |
| -------- | --------------------------------------------------------------------------------------- | ---------- |
| _1       | wepon特征工程中的基本特征 + 所有的other_feature + 线上特征 on_u1~on_u10, on_u12, on_u13 | 0.78687035 |
| _2       | 在 _1 的基础上，试一下 Andy 的调参参数                                                  | 0.79212071 |
| _3       | 使用所有特征，试一下 Andy 的调参参数                                                    | 0.79282404 |

## rf_preds

| versions | details                                                                                 | Score      |
| -------- | --------------------------------------------------------------------------------------- | ---------- |
| _1       | wepon特征工程中的基本特征 + 所有的other_feature + 线上特征 on_u1~on_u10, on_u12, on_u13 | 0.78096180 |
| _2       | 调参顺序：max_depth -> min_samples_split -> min_samples_leaf                            | 0.77985640 |

## stk_preds

| versions | details                      | Score      |
| -------- | ---------------------------- | ---------- |
| _1       | xgb_4 + gbdt_1 逻辑回归 C=10 | 0.50000000 |

## weighted

| versions | details                  | Score      |
| -------- | ------------------------ | ---------- |
| _1       | 0.65*xgb_4 + 0.35*gbdt_1 | 0.79242147 |