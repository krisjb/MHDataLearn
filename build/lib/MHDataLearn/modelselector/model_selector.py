import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


def split_data(X, Y):
    """
    This function split the features and the target into training and test set
    Params:
        X- (df containing predictors)
        y- (series conatining Target)
    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, y_train, X_test, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    return X_train, y_train, X_test, y_test


def reveal_best_classification_model(X_train, Y_train, X_test, Y_test):
    """
    This function build four classifcation models using four different Algorithm,
    -Logistic regression
    -Support Vecter Machines
    -Decision Tree Classifier
    -K Nearest Neighbors
    It runs a grid search cv through the models to determine the best hyper parameter and their best score.
    compares the scores of all the four Algorithms and creates a dictionary of their respective
    best score and best parameter as per the gridsearch cv hyperparameter tuning during cross validations.

    Params:
        X_train
        Y_train
        X_test
        Y_test
    Returns:
        A dataframe containing the four models with their best hyper parameter and best scores on both training and test data.

    """

    lr_parameters = {'C': [0.01, 0.1, 1],
                     'penalty': ['l2'], 'solver': ['lbfgs']}

    svm_paramters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
                     'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}

    tree_parameters = {'criterion': ['gini', 'entropy'],
                       'splitter': ['best', 'random'],
                       'max_depth': [2*n for n in range(1, 10)],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10]}

    knn_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}
    try:

        # creating and trainin a logistic model using a grid serach cv
        logreg = LogisticRegression()
        logreg_cv = GridSearchCV(logreg, lr_parameters, cv=10)
        logreg_cv.fit(X_train, Y_train)

        # creating and trainin a support vector machine model using a grid serach cv
        svm = SVC()
        svm_cv = GridSearchCV(svm, svm_paramters, cv=10)
        svm_cv.fit(X_train, Y_train)

        # creating and trainin a tree based model using a grid serach cv
        tree = DecisionTreeClassifier()
        tree_cv = GridSearchCV(tree, tree_parameters, cv=10)
        tree_cv.fit(X_train, Y_train)

        # creating and trainin Knearest neighbors model using a grid serach cv
        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, knn_parameters, cv=10)
        knn_cv.fit(X_train, Y_train)

    except Exception as e:
        print(F"Error {e}!")

    models_dict = {
        "models": ['loreg_cv', 'svm_cv', 'tree_cv', 'knn_cv'],
        "BestParams": [logreg_cv.best_params_, svm_cv.best_params_, tree_cv.best_params_, knn_cv.best_params_],
        "BestTraininScore": [logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_],
        "TestAcuracy": [logreg_cv.score(X_test, Y_test), svm_cv.score(X_test, Y_test),
                        tree_cv.score(X_test, Y_test), knn_cv.score(X_test, Y_test)]

    }
    performance_df = pd.DataFrame(models_dict)

    return performance_df
