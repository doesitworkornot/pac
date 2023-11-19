import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# POINT THE DIFFERENCE BETWEEN 9.1 AND 9.2??


def main():
    # Reading data
    df = pd.read_csv('data/lab9/titanic_prepared.csv')
    target = df.label
    feature = df.drop(['label'], axis=1)

    # Splitting on train, validation and test
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.1,
                                                                              random_state=1)

    tree_res = des_tree(feature_train, target_train, feature_test)
    acc(tree_res, target_test, 'Decision Tree')

    n_tree_res = des_tree_n(feature_train, target_train, feature_test, 1)
    acc(n_tree_res, target_test, 'N Decision Tree')  # Isn't magic happening here?

    xgboost_res = xgboost(feature_train, target_train, feature_test)
    acc(xgboost_res, target_test, 'XGBOOST')

    log_regression_res = log_regression(feature_train, target_train, feature_test)
    acc(log_regression_res, target_test, 'Logistic Regression')


def des_tree(feature_train, target_train, feature_test):
    model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    model.fit(feature_train, target_train)
    predict = model.predict(feature_test)
    return predict


def des_tree_n(feature_train, target_train, feature_test, n):
    model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    model.fit(feature_train, target_train)
    importances = model.feature_importances_
    features = feature_train.columns

    # Sorting arguments by n most important
    indices = np.argsort(importances)[::-1][:n]
    important_one = features[indices]
    new_train = feature_train[important_one]
    new_test = feature_test[important_one]

    # Retraining model
    return des_tree(new_train, target_train, new_test)


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


def acc(predict, target, alg):
    xor = np.logical_xor(predict, target)
    accuracy = (len(xor) - sum(xor)) / len(xor)
    print(f'{alg} accuracy is {accuracy}')


if __name__ == "__main__":
    main()
