from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes
import datetime

thresholds = [0.05, 0.1, 0.15]

# KNN
knn_dict = {'model': neighbors.KNeighborsClassifier(),
            'name': 'KNN',
            'params': {'metric': ['minkowski'],
                       'n_neighbors': [5],
                       'weights': ['uniform']}}

# Logistic regression
lr_dict = {'model': linear_model.LogisticRegression(),
           'name': 'Logistic regression',
           'params': {'penalty': ['l1', 'l2'],
                      'C': [1.0]}}

# Decision tree
dtree_dict = {'model': tree.DecisionTreeClassifier(),
              'name': 'Decision tree',
              'params': {'criterion': ['gini', 'entropy'],
                         'splitter': ['best'],
                         'max_depth': [None]}}

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
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [None],
                      'random_state': [100]}}               

# Bagging
bag_dict = {'model': ensemble.BaggingClassifier(),
            'name': 'Bagging',
            'params': {'base_estimator': [None],
                       'n_estimators': [10],
                       'max_samples': [1.0],
                       'max_features': [1.0],
                       'random_state': [100]}}

#Gradient Boosting
gboost_dict = {'model': ensemble.GradientBoostingClassifier(),
              'name': 'Gradient boosting',
              'params': {'loss': ['deviance'],
                         'learning_rate': [0.1],
                         'n_estimators': [100],
                         'max_depth': [3]}}

# Ada Boosting
aboost_dict = {'model': ensemble.GradientBoostingClassifier(),
               'name': 'ADA Boost',
               'params': {'loss': ['exponential'],
                          'learning_rate': [0.1],
                          'n_estimators': [100],
                          'max_depth': [3]}}

# Naive Bayes
bayes_dict = {'model': naive_bayes.GaussianNB(),
              'name': 'Gaussian Naive Bayes',
              'params': {}}

# List of all models
model_list = [lr_dict, dtree_dict, rf_dict, bag_dict, gboost_dict, aboost_dict, bayes_dict]

splits = [{'split_number': 0, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2001, 1, 1, 0, 0), 'test_start': datetime.datetime(2002, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2003, 1, 1, 0, 0)}, 
           {'split_number': 1, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2002, 1, 1, 0, 0), 'test_start': datetime.datetime(2003, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2004, 1, 1, 0, 0)}, 
           {'split_number': 2, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2003, 1, 1, 0, 0), 'test_start': datetime.datetime(2004, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2005, 1, 1, 0, 0)}, 
           {'split_number': 3, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2004, 1, 1, 0, 0), 'test_start': datetime.datetime(2005, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2006, 1, 1, 0, 0)}, 
           {'split_number': 4, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2005, 1, 1, 0, 0), 'test_start': datetime.datetime(2006, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2007, 1, 1, 0, 0)}, 
           {'split_number': 5, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2006, 1, 1, 0, 0), 'test_start': datetime.datetime(2007, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2008, 1, 1, 0, 0)}, 
           {'split_number': 6, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2007, 1, 1, 0, 0), 'test_start': datetime.datetime(2008, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2009, 1, 1, 0, 0)}, 
           {'split_number': 7, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2008, 1, 1, 0, 0), 'test_start': datetime.datetime(2009, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2010, 1, 1, 0, 0)}, 
           {'split_number': 8, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2009, 1, 1, 0, 0), 'test_start': datetime.datetime(2010, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2011, 1, 1, 0, 0)}, 
           {'split_number': 9, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2010, 1, 1, 0, 0), 'test_start': datetime.datetime(2011, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2012, 1, 1, 0, 0)}, 
           {'split_number': 10, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2011, 1, 1, 0, 0), 'test_start': datetime.datetime(2012, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2013, 1, 1, 0, 0)}, 
           {'split_number': 11, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2012, 1, 1, 0, 0), 'test_start': datetime.datetime(2013, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2014, 1, 1, 0, 0)}, 
           {'split_number': 12, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2013, 1, 1, 0, 0), 'test_start': datetime.datetime(2014, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2015, 1, 1, 0, 0)}, 
           {'split_number': 13, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2014, 1, 1, 0, 0), 'test_start': datetime.datetime(2015, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2016, 1, 1, 0, 0)}, 
           {'split_number': 14, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2015, 1, 1, 0, 0), 'test_start': datetime.datetime(2016, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2017, 1, 1, 0, 0)}, 
           {'split_number': 15, 'train_start': datetime.datetime(2000, 1, 1, 0, 0), 
           'train_end': datetime.datetime(2016, 1, 1, 0, 0), 'test_start': datetime.datetime(2017, 1, 1, 0, 0), 
           'test_end': datetime.datetime(2018, 1, 1, 0, 0)}]