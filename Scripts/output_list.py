'''
output_list.py

Produce list of facilities with actual and predicted target variable label
using a given model, parameters and test-train split. Output targeted
priority list of facilities for inspection, relative feature importances,
and precision-recall graph for best-performing model (determined by
the MODELTYPE and KWARGS globals)
'''

import sys
import math
import pandas as pd
import numpy as np
import generate_features as gf
import model_specs as ms
import pipeline as ppl
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, utils, base, preprocessing
from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes

# Can update the MODELTYPE and KWARGS arguments based on 
# the best model and specifications as determined by pick_best_model.py
MODELTYPE = ensemble.RandomForestClassifier()
KWARGS = {'criterion':'gini', 'n_estimators':10, 'max_depth':None, 'random_state':100}


def train_best_model(setup, splitnumber, modeltype, kwargs):
    '''
    Train a model using the best model type and specified parameters.

    Inputs:
      data (str): filepath of the input built dataset
      splitnumber (int): the number of the split from the 
        model_specs module
      modeltype (sklearn object): base model object
      kwargs (dict): keyword arguments to pass as parameters to model

    Returns:
      fitted model
    '''
    xtrain, ytrain = setup[splitnumber]['xtrain'], setup[splitnumber]['ytrain']
    xtest, ytest = setup[splitnumber]['xtest'], setup[splitnumber]['ytest']

    model = base.clone(modeltype)
    model.set_params(**kwargs)
    model.fit(xtrain, ytrain)
    return model


def identify_facilities(model, xtest, ytest, target, threshold):
    '''
    Takes a trained model object and returns a merged df of 
    the index nunmber from the original built dataset along with
    the 'actual' and 'predicted' target variable labels.

    Inputs:
      model: trained sklearn model object
      xtest (df): dataframe of feature testing data
      ytest (df): dataframe of target testing data
      target (str): name of target variable
      threshold (float): threshold at which to produce list of
        facilities

    Returns:
      merged (df): dataframe containing the indexes along with
        actual and predicted class labels for the trained model's
        predictions on a given test set
    '''
    pred_scores = model.predict_proba(xtest)
    ytest_reset = ytest.reset_index()

    pred_scores_binary = [x[1] for x in pred_scores]
    pred_scores_frame = pd.Series(pred_scores_binary).to_frame()
    pred_scores_frame.rename(columns={0:'proba'}, inplace=True)

    merged = ytest_reset.merge(
        pred_scores_frame, how='inner', left_index=True, right_index=True)
    merged.sort_values(by='proba', ascending=False, inplace=True)

    merged['rank'] = range(len(merged))
    merged['predicted'] = np.where(
        merged['rank']<math.floor(len(merged)*threshold), 1, 0)
    return merged


def feature_importances(train_test_dict, splitnumber, model):
    '''
    Rank features by their importance, output to CSV

    Inputs:
      train_test_dict (dict): dictionary containing train-test split
      model (sklearn obj): trained model object

    Returns:
      sorted list of features by importance (descending)
    '''
    out_df = pd.DataFrame(data={'names':train_test_dict[splitnumber]['xtest'].columns,
        'importances':model.feature_importances_})
    out_df = out_df.sort_values(by='importances', ascending=False)
    return out_df


def precision_recall_curve(train_test_dict, splitnumber, model, target):
    '''
    Produce summary df of precision and recall measures at all thresholds
    from 0 to 1 for a trained model.

    Inputs:
      train_test_dict (dict): dictionary of train-test splits with metadata
        and actual data for each split
      splitnumber (int): split ID number
      model (sklearn obj): trained sklearn model
      target (str): name of target variable

    Returns:
      summary df of precision and recall at every threshold from 0 - 1 by .01's
    '''
    summary_df = pd.DataFrame(columns=['threshold','precision','recall'])
    xtest = train_test_dict[splitnumber]['xtest']
    ytest = train_test_dict[splitnumber]['ytest']
    pred_scores = model.predict_proba(xtest)

    thresholds = np.arange(0.01,1,0.01)
    for threshold in thresholds:
        predicted, actual = ppl.get_pred_and_actual(
            xtest, ytest, target, pred_scores, threshold)
        precision = ppl.calculate_precision_at_threshold(actual, predicted)
        recall = ppl.calculate_recall_at_threshold(actual, predicted)

        summary_df.loc[len(summary_df)] = [threshold, recall, precision]

    return summary_df


def run_all(args):
    '''
    Run whole program: generate the test-train data on the specified
    split and train the model, then attach predicted class labels 
    to the testing data and output final list of predictions.

    Inputs: args (list of):
      data-infile (str): filepath to input built data set
      splitnumber (int): number of the desired analysis split from
        model_specs
      target-variable (str): name of the target class variable
      threshold (float): threshold level for assigning class labels
      full-outfile (str): filepath to output file for ALL rows (1's and 0's)
      targeted-outfile (str): filepath to output file for targeted
        facility inspection list (1's only)
      feature-importances-outfile (str): filepath to output list of
        features by relative importance

    Returns:
      Nothing, outputs full-outfile and targeted-outfile to CSV
    '''
    usage = ("python3 output_list.py <data-infile> <splitnumber>"
        " <target-variable> <threshold> <full-outfile> <targeted-outfile>"
        " <feature-importances-outfile> <precision-recall-curve-outfile>")

    if len(args) != 9:
        print(usage)
        sys.exit(1)

    # Create train-test split using the specified split number from model_specs
    splitnumber = int(args[2])
    setup = gf.clean_and_split(
        args[1], [ms.splits[splitnumber]], args[3])

    # Train model
    trained_model = train_best_model(setup, int(args[2]), MODELTYPE, KWARGS)

    # Create facility list of 1's and 0's from xtest data
    facility_list = identify_facilities(
        trained_model, setup[splitnumber]['xtest'], setup[splitnumber]['ytest'],
        args[3], float(args[4]))

    # Read in built data
    built_data = pd.read_csv('../Data/Built/full_built_data.csv')
    built_data = built_data.reset_index()

    # Merge built data with 1's and 0's from xtest and output to CSV
    outlist = built_data.merge(facility_list, on='index', how='inner')
    final_outlist = outlist[['year', 'ID_NUMBER', 'FACILITY_NAME', 'STREET_ADDRESS',
        'CITY_NAME', 'STATE_CODE', 'ZIP_CODE', 'violation', 'predicted']]
    final_outlist.rename(columns={'violation':'actual'}, inplace=True)
    final_outlist.to_csv(args[5])

    # Reduce list to only targeted facilities and output
    targeted_list = final_outlist[final_outlist['predicted'] == 1]
    targeted_outlist.to_csv(args[6])

    # Calculate feature importances and output
    f_imp = feature_importances(setup, splitnumber, trained_model)
    f_imp.to_csv(args[7])

    # Plot precision recall curve
    pr_df = precision_recall_curve(setup, 15, trained_model, 'violation')
    fig, ax = plt.subplots()
    summary_df.plot(ax=ax, kind='line', x='threshold', y='precision')
    summary_df.plot(ax=ax, kind='line', x='threshold', y='recall',
        xticks=(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0),
        yticks=(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0),
        title='Precision-recall curve for best-performing model in 2017 test set')
    plt.savefig(args[8])
    #plt.show()

    
if __name__ == "__main__":
    run_all(sys.argv)

