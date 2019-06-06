from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes
import datetime

train_test_splits = [(('01/01/1990','12/31/1994'),('01/01/1996','12/31/1996')),
                     (('01/01/2000','12/31/2008'),('01/01/2009','12/31/2012'))]
thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# KNN
knn_dict = {'model': neighbors.KNeighborsClassifier(),
            'name': 'KNN',
            'params': {'metric': ['minkowski'],
                       'n_neighbors': [50],
                       'weights': ['uniform']}}

# Logistic regression
lr_dict = {'model': linear_model.LogisticRegression(),
           'name': 'Logistic regression',
           'params': {'penalty': ['l1', 'l2'],
                      'C': [1.0, 0.75, 0.5, 0.1]}}

# Decision tree
dtree_dict = {'model': tree.DecisionTreeClassifier(),
              'name': 'Decision tree',
              'params': {'criterion': ['gini', 'entropy'],
                         'splitter': ['best', 'random'],
                         'max_depth': [1,2,3,5,10,None]}}

# SVM
svm_dict = {'model': svm.SVC(),
            'name': 'SVM',
            'params': {'C': [1.0],
                       'kernel': ['rbf'],
                       'max_iter': [5],
                       'probability':[True]}}

# Random forest
rf_dict = {'model': ensemble.RandomForestClassifier(),
           'name': 'Random forest',
           'params': {'n_estimators': [10, 50, 100],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1,2,3,5,10,None],
                      'random_state': [100]}}               

# Bagging
bag_dict = {'model': ensemble.BaggingClassifier(),
            'name': 'Bagging',
            'params': {'base_estimator': [linear_model.LogisticRegression()],
                       'n_estimators': [10, 50, 100],
                       'max_samples': [1, 5, 10],
                       'max_features': [1, 5, 10],
                       'random_state': [100]}}

#Boosting
boost_dict = {'model': ensemble.GradientBoostingClassifier(),
              'name': 'Gradient boosting',
              'params': {'loss': ['deviance', 'exponential'],
                         'learning_rate': [0.5, 0.01, 0.01],
                         'n_estimators': [50, 100, 200],
                         'max_depth': [1, 3, 10]}}

# Naive Bayes
bayes_dict = {'model': naive_bayes.GaussianNB(),
              'name': 'Gaussian Naive Bayes',
              'params': {}}

# List of all models
#model_list = [knn_dict, lr_dict, dtree_dict, svm_dict, rf_dict, bag_dict, boost_dict, bayes_dict]
model_list = [lr_dict, dtree_dict, rf_dict]



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