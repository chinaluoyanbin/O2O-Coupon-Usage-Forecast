import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(path):
    featureImportance = pd.read_csv(path, header=None)
    features = featureImportance.iloc[:, 0]
    importanceScore = featureImportance.iloc[:, 1]

    print(list(featureImportance.iloc[-19:, 0]))
    # print(featureImportance)

    f, ax = plt.subplots(figsize=(20, 5))
    sns.barplot(x=features, y=importanceScore)
    plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.bar(x, y)
    # plt.xlabel('Features')
    # plt.ylabel('Importance Score')
    # plt.title('Feature Importance')
    # plt.show()


if __name__ == '__main__':
    plot_feature_importance(path='./gbdt_featureImportance.csv')
