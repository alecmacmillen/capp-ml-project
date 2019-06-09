'''
pick_best_model.py
'''

import sys
import pandas as pd

INPUT_DTYPE = {'model':str, 'train_date':str, 'test_date':str,
               'parameters':str, 'threshold':float, 'baseline':float,
               'accuracy': float, 'precision':float, 'recall':float,
               'f1':float, 'auc':float}


def model_stats_by_year(infile, split, metric):
	'''
	'''
	info = pd.read_excel(infile, sheet_name=split, header=0, dtype=INPUT_DTYPE)
