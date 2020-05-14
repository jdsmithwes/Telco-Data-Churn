import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

def info(data):
    """ This function returns the shape and types of data in a dataframe"""
    print('The dataframe has a shape of:\n\n', data.shape)
    print('The dataframe has the following datatypes:\n\n', data.dtypes)
    print('The total number of each datatype is:\n\n\n', data.dtypes.value_counts())
    return data.head(5)
    

def null(data):
    """ This function provides a summary of the null values in the dataframe"""
    return print('Dataframe null values:', data,isnull().sum())
    
    

#def bestinputs(X,y)
 #   """ Highlights the most important features from dataset"""
  #  #borrowed from sckit's documentation
   # model = ExtraTreesClassifier(random_state=0)
    #model.fit(X,y)
    #importances = model.feature_importances_
    #std = np.std([model.feature_importances_ for tree in model.estimators_],axis=0)
    #indices = np.argsort(importances) [::-1]

    #Print feature ranking
    #print('Feature ranking:')

    #for f in range(X.shape[1]):
     #   print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    #Plot the feature importance of the forest
    #plt.figure()
    #plt.title('Feature Importance')
    #plt.bar(range(X.shape[1]),importances[indices], color = 'r', yerr=std[indices],align='center')
    #plt.xticks(range(X.shape[1]),indices)
    #plt.xlim([-1, X.shape[1]])
    #plt.show()
    #pass

def outliers(df):
    """ This function eliminates all outliers from a given dataframe with z-score above 3 threshold"""
    #borrowed from Kite website
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df[filtered_entries]
    return new_df.shape 

def results(y_test,y_model):
    """This function returns the confusion matrix and accuracy score of the model
    Inputs:
        y_test: provide actual data from data split
        y_model: provide the predicted results from model"""
        
    cm = confusion_matrix(y_test,y_model)
    acc = accuracy_score(y_test,y_model)
    print('The Accuracy Score for this model is {acc}'.format(acc=acc))
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(acc)
    plt.title('Confusion Matrix', size = 10)
    sns.set(font_scale=1.8)
    sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r'); 
    
    pass

def classification(y_test,y_model):
    """ This funtion returns the classification report"""
    return print(classification_report(y_test, y_model))
pass

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.ylabel('Features')
    plt.xlabel('Measure of Importance')
    plt.title('Feature Importance')
    plt.show()

# Specify your top n features you want to visualize.
# You can also discard the abs() function 
# if you are interested in negative contribution of features


