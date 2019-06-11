'''
pick_best_model.py
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
    '''
    info = pd.read_csv(infile, header=0, dtype=INPUT_DTYPE)
    idxmax = info.groupby(['model'])[metric].idxmax().tolist()
    maxes = info.iloc[idxmax]
    maxes.loc[:, 'test_year'] = maxes['test_date'].str.slice(6, 10)
    maxes = maxes[['test_year', 'model', metric]]
    return maxes


def all_model_stats(infilespath, metric, splitrange):
    '''
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
    plt.show()
    #plt.savefig(args[4])


if __name__ == "__main__":
    go(sys.argv)




