import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

train_data = pd.DataFrame.from_csv('trainData.csv')
Y = train_data['V1']
features = train_data.columns[1:]
X = train_data[features]

clf = GridSearchCV(GradientBoostingClassifier(), n_jobs=-1, cv = 20, scoring='accuracy', 
        param_grid={'loss':['exponential'], 'learning_rate':[0.02], 'n_estimators':np.arange(150, 401, 50), 
            'max_depth':[5], 'subsample':[0.7], 'max_features':[0.5], 'verbose':[1]})

''' 
Uncomment clf.fit to fit the final model on your own. But you'll save some time 
if you load the pre-trained models instead.
'''
#clf.fit(X, Y) 
originalClf = joblib.load("originalGridSearch/clf.pkl")
clf = joblib.load("nEstimatorsGridSearch/clf.pkl")

joblib.dump(clf, "finalModel/clf.pkl")
