# Higgs Competition, 2nd place
[Original competition documents](https://github.com/MachinesWhoLearn/competitions/tree/master/higgs)

##Methodology
The theme of the competition was "trees", and so I had experimented with various tree-based classifiers in R beforehand using the '''caret''' library. Ultimately I settled on using a stochastic gradient boosting model. All that was left to do was the hyperparameter tuning. But instead of splitting my data into a train and test set, I tuned upon the entire dataset - since I intended to train each individual tree within the SGB classifier on a randomly selected subset of the data available for training. This would allow me to obtain OOB error estimates and nullify the need for a separate test set.

There were two parts to my grid search process:

1. An all-encompassing grid search. This was meant to let me choose the "general" hyperparameters of my model. I defined a general hyperparameter to be any parameter that was not '''n_estimator''' (the number of trees, more on this in second item below). As a scoring metric, I used the area under an ROC curve (AUC) because I was not yet concerned with obtaining the best possible accuracy, I was only interested in obtaining a classifier (with some given set of general hyperparameters) that was most likely to perform well on holdout data. The optimal model had an AUC score of 0.784±0.015 and the following parameters:

    {'learning_rate': 0.02,
     'loss': 'exponential',
     'max_depth': 7,
     'max_features': 0.5,
     'n_estimators': 400,
     'subsample': 0.7}

But, ultimately, I decided to use a '''max_depth''' of 5 since it gave me a less complex model and an AUC score (0.782±0.014) within a standard error of the above parameter set.

2. An '''n_estimator'''s grid search. The number of trees was the most important variable in whether my model overfit the training data, and so I did a separate grid search to fine tune the '''n_estimators''' parameter (fixing the general parameters) and used accuracy as my scoring metric since I was to select my final classifier from the results of this second grid search. This gave me a model with an accuracy of 0.7074±0.0098 and the following parameter set: 

    {'learning_rate': 0.02,
     'loss': 'exponential',
     'max_depth': 5,
     'max_features': 0.5,
     'n_estimators': 250,
     'subsample': 0.7}

As you can see from the below plot, a second grid search was probably unnecessary, since I could have instead based my '''n_estimators''' choice by finding the peak in OOB error improvement as a function of '''n_estimators'''.

![OOB improvement](https://github.com/philerooski/mwl/blob/master/higgs/oob.png)

After choosing my final model, I predicted upon '''unlabeledTestSet''' - which was to determine the rankings of competitors - obtaining an accuracy of 0.7065, 0.0003 behind the [1st place winner](https://github.com/nelson-liu/ML-competitions/tree/master/MWL/Higgs).
