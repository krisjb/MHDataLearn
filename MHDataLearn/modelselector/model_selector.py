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


def default_model_variables(df):
    """
    Assigns default features to a list of predictor variables (X) -
        'age_admit', 'Gender', 'EthnicCategory', 'MaritalStatus', 
        'EmployStatus', 'SettledAccommodationInd', 'len_stay', 
        'MHCareClusterSuperClass', 'HospitalBedTypeMH', 
        'imd_dec' (if available)
    Assigns 'EmergencyReadmit' as the outcome variable (Y)
    
    Parameters
    ----------
    df : main dataset

    Returns
    -------
    X : default list of features for model (predictor variables)
    Y : default outcome variable (Emergency Readmission within 30 days)

    """
    Y = df['EmergencyReadmit']
    if 'imd_dec' in df.columns:
        feature_list = ['age_admit', 'Gender', 'EthnicCategory',
                        'MaritalStatus','EmployStatus', 
                        'SettledAccommodationInd', 'len_stay', 
                        'MHCareClusterSuperClass', 'HospitalBedTypeMH', 
                        'imd_dec'
                        ]
    else:
        feature_list = ['age_admit', 'Gender', 'EthnicCategory',
                        'MaritalStatus','EmployStatus', 
                        'SettledAccommodationInd', 'len_stay', 
                        'MHCareClusterSuperClass', 'HospitalBedTypeMH'
                        ]
    X = df[feature_list]
    return X, Y


def split_data(X, Y):
    """
    This function split the features and the target into training and test set
    Params:
        X- (df containing predictors)
        y- (series conatining Target)
    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    return X_train, X_test, Y_train, Y_test


# def reveal_best_classification_model(X_train, Y_train, X_test, Y_test):
#     """
#     This function build four classifcation models using four different Algorithm,
#     -Logistic regression
#     -Support Vecter Machines
#     -Decision Tree Classifier
#     -K Nearest Neighbors
#     It runs a grid search cv through the models to determine the best hyper parameter and their best score.
#     compares the scores of all the four Algorithms and creates a dictionary of their respective
#     best score and best parameter as per the gridsearch cv hyperparameter tuning during cross validations.

#     Params:
#         X_train
#         Y_train
#         X_test
#         Y_test
#     Returns:
#         A dataframe containing the four models with their best hyper parameter and best scores on both training and test data.

#     """

#     lr_parameters = {'C': [0.01, 0.1, 1],
#                      'penalty': ['l2'], 'solver': ['lbfgs']}

#     svm_paramters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
#                      'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}

#     tree_parameters = {'criterion': ['gini', 'entropy'],
#                        'splitter': ['best', 'random'],
#                        'max_depth': [2*n for n in range(1, 10)],
#                        'max_features': ['auto', 'sqrt'],
#                        'min_samples_leaf': [1, 2, 4],
#                        'min_samples_split': [2, 5, 10]}

#     knn_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                       'p': [1, 2]}
#     try:

#         # creating and trainin a logistic model using a grid serach cv
#         logreg = LogisticRegression()
#         logreg_cv = GridSearchCV(logreg, lr_parameters, cv=10)
#         logreg_cv.fit(X_train, Y_train)

#         # creating and trainin a support vector machine model using a grid serach cv
#         svm = SVC()
#         svm_cv = GridSearchCV(svm, svm_paramters, cv=10)
#         svm_cv.fit(X_train, Y_train)

#         # creating and trainin a tree based model using a grid serach cv
#         tree = DecisionTreeClassifier()
#         tree_cv = GridSearchCV(tree, tree_parameters, cv=10)
#         tree_cv.fit(X_train, Y_train)

#         # creating and trainin Knearest neighbors model using a grid serach cv
#         KNN = KNeighborsClassifier()
#         knn_cv = GridSearchCV(KNN, knn_parameters, cv=10)
#         knn_cv.fit(X_train, Y_train)

#     except Exception as e:
#         print(F"Error {e}!")

#     models_dict = {
#         "models": ['loreg_cv', 'svm_cv', 'tree_cv', 'knn_cv'],
#         "BestParams": [logreg_cv.best_params_, svm_cv.best_params_, tree_cv.best_params_, knn_cv.best_params_],
#         "BestTraininScore": [logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_],
#         "TestAcuracy": [logreg_cv.score(X_test, Y_test), svm_cv.score(X_test, Y_test),
#                         tree_cv.score(X_test, Y_test), knn_cv.score(X_test, Y_test)]

#     }
#     performance_df = pd.DataFrame(models_dict)

#     return performance_df


def reveal_best_classification_model(X_train, Y_train, X_test, Y_test):
    """
    This function build four classifcation models using four different Algorithm,
    -Logistic regression
    -Support Vecter Machines
    -Decision Tree Classifier
    -K Nearest Neighbors
    It runs a grid search cv through the models to determine the best hyper parameter and their best score.
    compares the scores of all the four Algorithms and creates a dictionary of the model with their respective 
    best score and best parameter as per he gridsearch cv hyperparameter tuning during cross validations.
    
    Params:
        X_train
        Y_train
        X_test
        Y_test
    Returns:
        A dataframe containing the four models with their best hyper parameter and best scores.
    
    """
    lr_parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}
    
    tree_parameters =  {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}
    
    knn_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
    try:
        
        #creating and trainin a logistic model using a grid serach cv
        logreg = LogisticRegression()
        logreg_cv = GridSearchCV(logreg, lr_parameters, cv=10)
        logreg_cv.fit(X_train, Y_train)

        #creating and trainin a support vector machine model using a grid serach cv
        # svm = SVC()
        # svm_cv = GridSearchCV(svm, svm_paramters, cv=5, refit=True)
        # svm_cv.fit(X_train, Y_train)

        #creating and trainin a tree based model using a grid serach cv            
        tree = DecisionTreeClassifier()
        tree_cv = GridSearchCV(tree, tree_parameters, cv=10)
        tree_cv.fit(X_train, Y_train)

        #creating and trainin Knearest neighbors model using a grid serach cv
        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, knn_parameters, cv=10)
        knn_cv.fit(X_train, Y_train)
        
    except Exception as e:
        print(F"Error {e}!")
    print(logreg_cv)
    models_dict = {
        "models":['logreg_cv', 'tree_cv', 'knn_cv'],
        "BestParams":[logreg_cv.best_params_, tree_cv.best_params_, knn_cv.best_params_],
        "BestTraininScore":[logreg_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_],
        "TestAcuracy": [logreg_cv.score(X_test, Y_test), tree_cv.score(X_test, Y_test), knn_cv.score(X_test, Y_test)]

    }
    performance_df = pd.DataFrame(models_dict)
        
    return performance_df


def visualize_model_performance(reveal_best_classification_model):
    """
    This function creates a bar plot with score of the individual models and their labels.
    Params:
        reveal_best_classification_model type(function)
    Output:
        Barplot
    """
    
    
    return reveal_best_classification_model.plot(kind="bar")
    
    
    
    
    
def plot_confusion_matrix(yhat,y_predict):
    """
    This function plots the confusion matrix of the models.
    
    Params:
        Y_test
        Predictions
    Output:
        Confusion matrix
    
    
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Emegency_Readmit', 'Not Emergency Readmit']); ax.yaxis.set_ticklabels(['Emegency_Readmit', 'Not Emergency Readmit'])



def train_default_model(df, default_features=True):
    """
    Splits a dataset, trains classification models on the training dataset and 
    returns test dataset performance metrics for each model.
    Allows use of a default feature set, or if a bespoke feature set or 
    outcome variable has already been defined, runs the model using these
    features and outcome variables

    Parameters
    ----------
    df : main dataset
    default features : Boolean
        If True, the model will be trained to predict emergency readmissions
        based on a default features set (see function default_model_variables)
        If False, ensure you have defined your own variables X and Y,
            where X is a dataframe of feature variables (predictors) and
            Y is a dataframe of the outcome variable.
        The default is TRUE

    Returns
    -------
    performance_df : a dataframe of performance metrics for the best
                        machine learning classification model

    """
    if default_features:
        df = default_model_variables(df)
        X_train, Y_train, X_test, Y_test = split_data(X, Y)
        reveal_best_classification_model(X_train, Y_train, X_test, Y_test)
    else:
        X_train, Y_train, X_test, Y_test = split_data(X, Y)
        reveal_best_classification_model(X_train, Y_train, X_test, Y_test)
    print(f'The performance metrics for the best model are: {performance_df}')
    visualize_model_performance(performance_df)
    