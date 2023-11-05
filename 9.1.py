import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def acc(predict, target, alg):
    xor = np.logical_xor(predict, target)
    accuracy = (len(xor) - sum(xor)) / len(xor)
    print(f'{alg} accuracy is {accuracy}')


def forest(feature_train, target_train, feature_val):
    # Training RandomForest
    model = RandomForestClassifier(n_estimators=10, max_depth=4, criterion='entropy')
    model.fit(feature_train, target_train)
    predict = model.predict(feature_val)
    # importance = model.feature_importances_
    # columns = feature_train.columns
    # for i in range(len(columns)):
    #     print(f'Importance of feature {columns[i]} is {importance[i]*100}%')
    return predict


def xgboost(feature_train, target_train, feature_val):
    model = XGBClassifier(n_estimators=20, max_depth=4)
    model.fit(feature_train, target_train)
    predict = model.predict(feature_val)
    return predict


def log_regression(feature_train, target_train, feature_val):
    model = LogisticRegression(C=0.1, solver='lbfgs')
    model.fit(feature_train, target_train)
    predict = model.predict(feature_val)
    return predict


def knn(feature_train, target_train, feature_val):
    model = KNeighborsClassifier()
    model.fit(feature_train, target_train)
    predict = model.predict(feature_val)
    return predict


def main():
    # Reading data
    df = pd.read_csv('data/lab9/train.csv')
    df.Sex.replace(['male', 'female'], [1, 0], inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    target = df.Survived
    feature = df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Splitting on train, validation and test
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2,
                                                                              random_state=1)
    feature_train, feature_val, target_train, target_val = train_test_split(feature_train, target_train, test_size=0.2,
                                                                            random_state=1)
    forest_res = forest(feature_train, target_train, feature_val)
    acc(forest_res, target_val, 'Random Forest')

    xgboost_res = xgboost(feature_train, target_train, feature_val)
    acc(xgboost_res, target_val, 'XGBOOST')

    log_regression_res = log_regression(feature_train, target_train, feature_val)
    acc(log_regression_res, target_val, 'Logistic Regression')

    knn_res = knn(feature_train, target_train, feature_val)
    acc(knn_res, target_val, 'k Nearest Neighbours')


if __name__ == "__main__":
    main()
