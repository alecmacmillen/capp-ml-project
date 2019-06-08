'''
run_models
'''

import sys
import pandas as pd
import merge_and_collapse as mc
import generate_features as gf
import pipeline as ppl
import model_specs as ms


def go(args):
    '''
    Run all models using specifications and train-test splits 
    listed in model_specs_short
    '''
    usage = ("usage: python3 run_models.py <infile.csv> <outfile.xlsx>")
    if len(args) != 3:
        print(usage)
        sys.exit(1)


    # Create dictionary of train-test sets and output Excel writer object
    train_test_dict = gf.clean_and_split(args[1], ms.splits, 'violation')
    writer = pd.ExcelWriter(args[2], engine='xlsxwriter')

    # Iterate through all train-test sets
    for split in train_test_dict:
        print("Running models for split ", split, " for train dates ", 
            train_test_dict[split]['train_dates'], " and test dates ", 
            train_test_dict[split]['test_dates'])

        # Unpack feature and target train-test sets
        xtrain, ytrain = train_test_dict[split]['xtrain'], train_test_dict[split]['ytrain']
        xtest, ytest = train_test_dict[split]['xtest'], train_test_dict[split]['ytest']

        # Unpack train-test date strings
        train_dates = train_test_dict[split]['train_dates'][0] + "-" + \
            train_test_dict[split]['train_dates'][1]
        test_dates = train_test_dict[split]['test_dates'][0] + "-" + \
            train_test_dict[split]['test_dates'][1]
        
        # Call model 'magic loop' from pipeline module
        split_summary = ppl.run_all_models(
            xtrain, ytrain, xtest, ytest, train_dates, test_dates, 'violation', ms.model_list, 'precision')

        # Write out model results to output sheet
        sheetname = 'split_' + str(split)
        split_summary.to_excel(writer, sheet_name=sheetname, header=True, index=False)
        writer.save()

    writer.save()


if __name__ == "__main__":
    go(sys.argv)



