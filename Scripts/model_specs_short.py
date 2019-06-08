from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes

train_test_splits = [(('01/01/1990','12/31/1994'),('01/01/1996','12/31/1996')),
                     (('01/01/2000','12/31/2008'),('01/01/2009','12/31/2012'))]

thresholds = [0.01, 0.5]

# KNN
knn_dict = {'model': neighbors.KNeighborsClassifier(),
            'name': 'KNN',
            'params': {'metric': ['euclidean'],
                       'n_neighbors': [25],
                       'weights': ['distance']}}

# Logistic regression
lr_dict = {'model': linear_model.LogisticRegression(),
           'name': 'Logistic regression',
           'params': {'penalty': ['l2'],
                      'C': [1.0]}}

# Decision tree
dtree_dict = {'model': tree.DecisionTreeClassifier(),
              'name': 'Decision tree',
              'params': {'criterion': ['gini'],
                         'splitter': ['best'],
                         'max_depth': [10]}}

# SVM
svm_dict = {'model': svm.SVC(),
            'name': 'SVM',
            'params': {'C': [1.0],
                       'kernel': ['rbf'],
                       'max_iter': [-1],
                       'probability':[True]}}

# Random forest
rf_dict = {'model': ensemble.RandomForestClassifier(),
           'name': 'Random forest',
           'params': {'n_estimators': [10],
                      'criterion': ['gini'],
                      'max_depth': [2],
                      'random_state': [100]}}               

# Bagging
bag_dict = {'model': ensemble.BaggingClassifier(),
            'name': 'Bagging',
            'params': {'base_estimator': [linear_model.LogisticRegression()],
                       'n_estimators': [10],
                       'max_samples': [2],
                       'max_features': [1],
                       'random_state': [100]}}

#Boosting
boost_dict = {'model': ensemble.GradientBoostingClassifier(),
              'name': 'Gradient boosting',
              'params': {'loss': ['deviance', 'exponential'],
                         'learning_rate': [0.5],
                         'n_estimators': [10],
                         'max_depth': [3]}}

# Naive Bayes
bayes_dict = {'model': naive_bayes.GaussianNB(),
              'name': 'Gaussian Naive Bayes',
              'params': {}}

# List of all models
model_list = [knn_dict, lr_dict, dtree_dict, svm_dict, rf_dict, bag_dict, boost_dict, bayes_dict]