'''
pick_best_model.py

Pick the best-performing model in each year and overall according
to the performance splits output by run_models.py
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DTYPE = {'model':str, 'train_date':str, 'test_date':str,
               'parameters':str, 'threshold':float, 'baseline':float,
               'accuracy': float, 'precision':float, 'recall':float,
               'f1':float, 'auc':float}

COLOR_DICT = {'KNN':'orange', 'Logistic regression':'red', 'Decision tree':'green',
              'SVM':'yellow', 'Random forest':'blue', 'Bagging':'purple',
              'Gradient boosting':'gray', 'ADA Boost':'brown', 
              'Gaussian Naive Bayes':'pink'}


def model_stats_by_year(infile, metric):
    '''
    Calculate model statistics by year. Read in a CSV of all model
    metrics for a given split (year) and identify the best-performing
    model of each type to plot.

    Inputs:
      infile (str): file path to input CSV of model metrics for a given year
      metric (str): metric to prioritize/maximize for (e.g. 'precision')

    Returns:
      maxes (pd df): df of the best-performing model of each type and its
        value for the prioritized metric for each year
    '''
    info = pd.read_csv(infile, header=0, dtype=INPUT_DTYPE)
    idxmax = info.groupby(['model'])[metric].idxmax().tolist()
    maxes = info.iloc[idxmax]
    maxes.loc[:, 'test_year'] = maxes['test_date'].str.slice(6, 10)
    maxes = maxes[['test_year', 'model', metric]]
    return maxes


def all_model_stats(infilespath, metric, splitrange):
    '''
    Wrapper function for model_stats_by_year that calculates all model
    stats for all years in the data.

    Inputs:
      infilespath (str): path to location where model metrics are saved in CSVs
      metric (str): metric to prioritize (e.g. 'precision')
      splitrange (int): number of highest split to consider

    Returns:
      summary df of best-performing model according to priority metric by
        split/year.
    '''
    summary = pd.DataFrame(columns=['test_year', 'model', metric])
    for i in range(splitrange):
        infile = infilespath + str(i) + '_models.csv'
        yearly = model_stats_by_year(infile, metric)
        summary = summary.append(yearly)
    summary.loc[:, 'test_year'] = summary['test_year'].astype(int)
    summary = summary.sort_values(by=['model', 'test_year'])
    summary.loc[:, 'color'] = summary['model'].map(COLOR_DICT)
    return summary


def count_best_years(summary, metric):
    '''
    Count the number of years for which a given model was the "best performing."

    Inputs:
      summary (pd df): pandas dataframe of by-model-type summary of year
        and performance.
      metric (str): metric to prioritize

    Returns:
      Nothing, prints number of splits/years each type of model was "best performing"
    '''
    summary = summary.reset_index()
    idxmax = summary.groupby(['test_year'])[metric].idxmax().tolist()
    maxes = summary.iloc[idxmax]
    max_dict = {}
    for row in maxes.iterrows():
        max_dict[row[1]['model']] = max_dict.get(row[1]['model'], 0) + 1
    print(max_dict)


def go(args):
    '''
    Print best-performing model counts and plot best-performing model
    by priority metric for all testing years.
    '''
    stats_df = all_model_stats(args[1], args[2], args[3])

    count_best_years(stats_df)

    fig, ax = plt.subplots()
    for key, grp in stats_df.groupby(['model']):
        ax = grp.plot(ax=ax, kind='line', x='test_year', 
            y=args[2], c=COLOR_DICT[key], label=key,
            xticks=range(min(grp['test_year']), max(grp['test_year']+1)),
            ylim=(0,1), yticks=(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0),
            title='Plot of best-performing model of each type by ' + args[2] + ' and year')

    ax.set_xlabel('Year of test period')
    ax.set_ylabel(args[2])
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(args[4])


if __name__ == "__main__":
    go(sys.argv)




