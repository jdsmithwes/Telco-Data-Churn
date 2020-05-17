#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:52:23 2020

@author: jamaalsmith
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def preprocess(X, y):
    '''Takes in features and target and implements all preprocessing steps for categorical and continuous features returning 
    train and test DataFrames with targets'''
    
    # Train-test split (75-25), set seed to 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    
    # Remove "object"-type features and SalesPrice from X
    cont_features = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]

    X_train_cont = X_train.loc[:, cont_features]
    X_test_cont = X_test.loc[:, cont_features]

    # Impute missing values with median using SimpleImputer
    impute = SimpleImputer(strategy='median')

    global X_train_imputed = impute.fit_transform(X_train_cont)
    global X_test_imputed = impute.transform(X_test_cont)

    # Scale the train and test data
    ss = StandardScaler()

    global X_train_imputed_scaled = ss.fit_transform(X_train_imputed)
    global X_test_imputed_scaled = ss.transform(X_test_imputed)

    # Create X_cat which contains only the categorical variables
    features_cat = [col for col in X.columns if X[col].dtype in [np.object]]
    global X_train_cat = X_train.loc[:, features_cat]
    global X_test_cat = X_test.loc[:, features_cat]

    # Fill nans with a value indicating that that it is missing
    global X_train_cat.fillna(value='missing', inplace=True)
    global X_test_cat.fillna(value='missing', inplace=True)

    # OneHotEncode Categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore')

    global X_train_ohe = ohe.fit_transform(X_train_cat)
    global X_test_ohe = ohe.transform(X_test_cat)

    columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
    cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
    
    # Combine categorical and continuous features into the final dataframe
    global X_train_all = pd.concat([pd.DataFrame(X_train_imputed_scaled), cat_train_df], axis=1)
    global X_test_all = pd.concat([pd.DataFrame(X_test_imputed_scaled), cat_test_df], axis=1)
    
    return X_train_all, X_test_all, y_train, y_test

def optimal_alpha(X,y):
    """Returns the best alpha level hyperparameter for your model"""
    X_train_all, X_test_all, y_train, y_test = preprocess(X, y)

    train_mse = []
    test_mse = []
    alphas = []

    for alpha in np.linspace(0, 200, num=50):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_all, y_train)
    
        train_preds = lasso.predict(X_train_all)
        train_mse.append(mean_squared_error(y_train, train_preds))
    
        test_preds = lasso.predict(X_test_all)
        test_mse.append(mean_squared_error(y_test, test_preds))
    
        alphas.append(alpha)
    
    import matplotlib.pyplot as plt
   

    fig, ax = plt.subplots()
    ax.plot(alphas, train_mse, label='Train')
    ax.plot(alphas, test_mse, label='Test')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('MSE')

    # np.argmin() returns the index of the minimum value in a list
    optimal_alpha = alphas[np.argmin(test_mse)]

    # Add a vertical line where the test MSE is minimized
    ax.axvline(optimal_alpha, color='black', linestyle='--')
    ax.legend();

    print(f'Optimal Alpha Value: {int(optimal_alpha)}')

def reg_simulation(n, random_state):
    X, y = make_regression(n_samples=100, n_features=1, noise=n, random_state=random_state)

    plt.scatter(X[:, 0], y, color='red', s=10, label='Data')

    reg = LinearRegression().fit(X, y)
    plt.plot(X[:, 0], reg.predict(X), color='black', label='Model')
    plt.title('Noise: ' + str(n) + ', R-Squared: ' + str(round(reg.score(X,y), 2)))
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
    plt.legend()
    plt.show()

random_state = np.random.RandomState(42)


for n in [10, 25, 40, 50, 100, 200]:
    reg_simulation(n, random_state)
    

def conf_matrix(y_true, y_pred):
    """Produces an array for your confusion matrix"""
    cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    for ind, label in enumerate(y_true):
        pred = y_pred[ind]
        if label == 1:
            # CASE: TP 
            if label == pred:
                cm['TP'] += 1
            # CASE: FN
            else:
                cm['FN'] += 1
        else:
            # CASE: TN
            if label == pred:
                cm['TN'] += 1
            # CASE: FP
            else:
                cm['FP'] += 1
    plt.imshow(conf_matrix,  cmap=plt.cm.Blues) 

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add appropriate axis scales
    class_names = set(y) # Get class labels to add to matrix
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add labels to each cell
    thresh = conf_matrix.max() / 2. # Used for text coloring below
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

    # Add a legend
    plt.colorbar()
    plt.show()
