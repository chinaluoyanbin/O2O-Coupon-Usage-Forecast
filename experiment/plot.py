import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(path):
    featureImportance = pd.read_csv(path, header=None)
    # features = featureImportance.iloc[:, 0]
    # importanceScore = featureImportance.iloc[:, 1]

    print(list(featureImportance.iloc[-19:, 0]))
    # print(featureImportance)

    # f, ax = plt.subplots(figsize=(6, 15))
    # sns.barplot(x=importanceScore, y=features)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.bar(x, y)
    # plt.xlabel('Features')
    # plt.ylabel('Importance Score')
    # plt.title('Feature Importance')
    # plt.show()


if __name__ == '__main__':
    plot_feature_importance(path='./gbdt_featureImportance.csv')

    # 删除19个
    drop_columns = [
        'u26', 'm21', 'on_u6', '4', '7', 'on_u8', 'u21', 'u31', '2', '6',
        'u16', '1', 'u33', 'u32', '5', '3', 'on_u10', 'on_u12'
        'on_u9'
    ]
    prediction(model='gbdt', drop_columns=drop_columns)
