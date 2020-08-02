import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, tree, model_selection
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv", delimiter=",")
test_X = pd.read_csv("test.csv", delimiter=",")
check_y = pd.read_csv("submission3.csv", delimiter=",")
check_y = check_y["Survived"].values
# test = pd.merge(test_y, test_X, on="PassengerId")

def data_cleaner(df):
    df["Age"] = df["Age"].fillna(df["Age"].dropna().median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().mean())

    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].dropna().mode()[0])
    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "C", "Embarked"] = 1
    df.loc[df["Embarked"] == "Q", "Embarked"] = 2

data_cleaner(train)
data_cleaner(test_X)
# print(train.count(), test_X.count())

train_X = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
train_y = train["Survived"].values
poly = preprocessing.PolynomialFeatures(degree=2)
poly_train_X = poly.fit_transform(train_X)
test_X = test_X[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
poly_test_X = poly.fit_transform(test_X)

def log_reg(train_X, poly_train_X, train_y, test_X, poly_test_X):
    clf1 = linear_model.LogisticRegression(max_iter=4000)
    fitted_clf1 = clf1.fit(train_X, train_y)

    clf2 = linear_model.LogisticRegression(max_iter=4000)
    fitted_clf2 = clf2.fit(poly_train_X, train_y)

    print(f"[Linear]\nTrain acc: {fitted_clf1.score(train_X, train_y)}")
    print(f"[2nd Order Poly]\nTrain acc: {fitted_clf2.score(poly_train_X, train_y)}")

def tree_clf(train_X, train_y, test_X):
    tree_clf = tree.DecisionTreeClassifier(random_state=1, criterion="entropy", splitter="random",
                                                max_depth=5, min_samples_split=3)
    fitted_tree_clf = tree_clf.fit(train_X, train_y)
    cross_val = model_selection.cross_val_score(fitted_tree_clf, train_X, train_y, cv=10)

    print(f"[Tree Classifier]\nTrain acc: {tree_clf.score(train_X, train_y)}")
    print(f"Cross validation: {cross_val}\nMean: {cross_val.mean()}")
    print(f"Similarity to last submission: {tree_clf.score(test_X, check_y)}")

    prediction = tree_clf.predict(test_X)
    return prediction

def gridsearch_tree(train_X, train_y):
    treeclf_para_grid = {"criterion": list(("gini", "entropy")), "splitter": list(("best", "random")),
                         "max_depth": list((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
                         "min_samples_split": list((2, 3, 4, 5, 6, 7))}
    grid = model_selection.GridSearchCV(tree.DecisionTreeClassifier(random_state=1),
                                        param_grid=treeclf_para_grid, cv=10)
    grid.fit(train_X, train_y)
    print(f"[Grid Search Decision Tree]\n{grid.best_params_}\nBest acc: {grid.best_score_}")

def rand_forest(train_X, train_y, test_X):
    rf_clf = RandomForestClassifier(random_state=1, criterion="entropy",
                                    max_depth=5, min_samples_split=3)
    fitted_rf_clf = rf_clf.fit(train_X, train_y)
    cross_val = model_selection.cross_val_score(fitted_rf_clf, train_X, train_y, cv=10)

    print(f"[Random Forest Classifier]\nTrain acc: {rf_clf.score(train_X, train_y)}")
    print(f"Cross validation: {cross_val}\nMean: {cross_val.mean()}")
    print(f"Similarity to last submission: {rf_clf.score(test_X, check_y)}")

    prediction = rf_clf.predict(test_X)
    return prediction

def gridsearch_rf(train_X, train_y):
    rfclf_para_grid = {"criterion": list(("gini", "entropy")),
                         "max_depth": list((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
                         "min_samples_split": list((2, 3, 4, 5, 6, 7))}
    grid = model_selection.GridSearchCV(RandomForestClassifier(random_state=1),
                                        param_grid=rfclf_para_grid, cv=10)
    grid.fit(train_X, train_y)
    print(f"[Grid Search Random Forest]\n{grid.best_params_}\nBest acc: {grid.best_score_}")

# log_reg(train_X, poly_train_X, train_y, test_X, poly_test_X)
# result = tree_clf(train_X, train_y, test_X)
# gridsearch_tree(train_X, train_y)
result = rand_forest(train_X, train_y, test_X)
# gridsearch_rf(train_X, train_y)
#
# with open("submission5.csv", "w") as fp:
#     fp.write("PassengerId,Survived\n")
#     for i, guess in enumerate(result):
#         fp.write(f"{892+i},{guess}\n")

