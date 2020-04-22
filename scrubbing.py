import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats

def info(data):
    return data.dtypes, data.shape, data.head()
pass

def bestinputs(X,y):
    #borrowed from sckit's documentation
    model = ExtraTreesClassifier(random_state=0)
    model.fit(X,y)
    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances) [::-1]

    #Print feature ranking
    print('Feature ranking:')

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    #Plot the feature importance of the forest
    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]),importances[indices], color = 'r', yerr=std[indices],align='center')
    plt.xticks(range(X.shape[1]),indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
pass

def outliers(df):
    #borrowed from Kite website
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df[filtered_entries]
    return new_df.shape    

    


