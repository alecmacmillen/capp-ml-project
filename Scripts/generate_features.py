'''
generate_features.py.

Clean combined EPA RCRA data and generate features in preparation for
using it to fit models in sklearn.
'''

import sys
import math
import pandas as pd
import numpy as np
import merge_and_collapse as mc
import utils
from sklearn import preprocessing


def strip_whitespace(df, cols):
    '''
    Strip whitespace from character columns.
    
    Inputs: 
      df (pandas df): dataframe containing columns to strip
      cols (list of str): list of columns to strip whitespace

    Returns:
      cleaned df with whitespace stripped from specified cols
    '''
    for col in cols:
        df.loc[:, col] = df[col].str.strip()
    return df


def dummify_categorical(df, cols, prefix=None):
    '''
    Dummify categorical variables and replace the original column in the
    dataframe with the corresponding dummy columns

    Inputs:
      df (pandas df): dataframe containing columns to dummify
      col (str): column to convert to dummies
      prefix (str, optional): prefix to append to beginning
        dummified variable name

    Returns:
      cleaned df with specified variable dummified
    '''
    dummified = pd.get_dummies(df, columns=cols, prefix=prefix)
    return dummified


def create_indicator_for_contains(df, oldcol, newcol, char):
    '''
    Create a newcol = 1 if 'char' contained somewhere in oldcol,
    else 0.

    Inputs:
      df (pandas df): dataframe containing column to convert
      oldcol (str): existing column to scan
      newcol (str): name of new binary column that takes 1 if
        oldcol contains characters from var 'char', else 0
      char (str): character expression to scan for in oldcol

    Returns:
      cleaned df with specified variable turned to indicator
    '''
    df[newcol] = np.where(df[oldcol].str.contains(char), 1, 0)
    return df


def impute_missing(df, col, fill):
    '''
    Impute missing values in a column with specified fill value.

    Inputs:
      df (pandas df): dataframe containing column to impute
      col (str): name of column to impute
      fill: fill value

    Returns:
      cleaned df with specified variable filled with fillvalue
    '''
    df[col].fillna(fill, inplace=True)
    return df


def indicator_timeframe(
    df, colname, colname_short, eval_datecol, prev_datecol):
    '''
    Creates an indicator variable that takes a value of 1 if the
    specified 'prev_datecol' falls within a series of intervals of
    eval_date (for example, is the most recent violation within
    6, 12, etc... months of the current eval_date?).

    Inputs:
      df (pandas df): dataframe containing column to calculate intervals
      colname (str): name of new column that shows timedelta in months
        between most recent occurrence of a given thing (e.g. violation)
        and the eval_date of the current observation
      colname_short (str): informative prefix of the new binary indicator
        columns
      eval_datecol (str): name of field containing evaluation date
      prev_datecol (str): name of field containing date of the previous
        "thing" to calculate timedelta

    Returns:
      cleaned df with specified variable converted to binary indicators
        for whether occurrence present in past 6,12,18,24,60,120 months
    '''
    df[colname] = 12 * (df[eval_datecol].dt.year - df[prev_datecol].dt.year) + (
        df[eval_datecol].dt.month - df[prev_datecol].dt.month)

    for month in [6,12,18,24,60,120]:
        newcol = colname_short + "_" + str(month) +"mo"
        df[newcol] = np.where(df[colname] <= month, 1, 0)
    return df


def scaler(df, col):
    '''
    Scales numeric columns using min-max scaling. Handles exception that
    occurs when all vals in a column are 0 by passing all 0's to the
    scaled column output.

    Inputs:
      df (pandas df): dataframe containing column to scale
      col (str): name of numeric column to scale

    Returns:
      cleaned df with specified variable scaled using min-max scaling
    '''
    maximum = max(df[col])
    minimum = min(df[col])
    try:
        df[col] = df[col].apply(lambda x: (x - minimum)/(maximum - minimum))
    except ZeroDivisionError:
        df[col] = 0
    return df


def fill_missing_cols(df1, df2):
    '''
    Loop through column lists for train and test splits to make sure they both
    have the same number of columns. (For example, if one of the datasets had
    a value for a categorical variable that the other didn't, the dummified
    column would be missing in the dataframe without that value.) For any 
    columns that don't exist, create it and set it to 0 for all values.

    Inputs:
      df1, df2 (pandas dfs): train-test split dataframes

    Returns:
      same dataframes with columns standardized
    '''
    for col in utils.features:
        if col not in df1.columns:
            df1.loc[:, col] = 0
        if col not in df2.columns:
            df2.loc[:, col] = 0

    df1 = df1[utils.features]
    df2 = df2[utils.features]

    return df1, df2


def clean_and_split(infile, train_test_list, target):
    '''
    Wrapper function for prep_features function. Takes a CSV file to read in,
    a list of tuples that show train/test dates, and the name of the target
    variable. Splits the dataset, generates features as per prep_features,
    and stores the resulting train-test feature-target splits in a dictionary.
    Can handle multiple train-test splits in one call.

    Inputs:
      infile (str): filepath specifying input CSV
      train_test_list (list of tuples): list of tuples that manually delineate
        the train-test dates. Each element of the list should take the form:
        ((train_start,train_end),(test_start,test_end))
      target (str): name of target prediction variable

    Returns:
      train_test_dict (dict): dictionary where keys are integers (counter var),
        values are nested dictionary of train/test split including train dates,
        test dates, feature/target training sets, and feature/target testing sets 
    '''
    # Create return dictionary; read in raw merged data and drop extraneous cols
    train_test_dict = {}
    df = pd.read_csv(infile, header=0, dtype=utils.INPUT_DTYPE)
    for col in utils.DROPCOLS:
        try:
            df = df.drop(columns=col)
        except KeyError:
            continue

    # Convert columns with date values to dates
    datecols = ['plan_date', 'eval_date', 'most_recent_viol', 'most_recent_enf']
    yrmonthcols = []
    for col in datecols:
        df[col] = pd.to_datetime(df[col])

    # Run loop once for each train/test set enumerated in train_test_list
    i = 0
    for split in train_test_list:
        print("Creating split ", i, " of ", len(train_test_list)-1)
        # Unpack train/test dates
        train_start = split['train_start']
        train_end = split['train_end']
        test_start = split['test_start']
        test_end = split['test_end']
        split_num = split['split_number']

        traindate_string = train_start.strftime('%m/%d/%Y') + ' - ' + \
            train_end.strftime('%m/%d/%Y')

        testdate_string = test_start.strftime('%m/%d/%Y') + ' - ' + \
            test_end.strftime('%m/%d/%Y')

        # Split dataset into train and test for separate preparation
        # Use the planning date (1/1 of each year) as the cutoff for train-test
        train = df[(df['plan_date']>=train_start) & (df['plan_date']<train_end)]
        test = df[(df['plan_date']>=test_start) & (df['plan_date']<test_end)]

        # Call prep_features function (see below)
        train_prepped = prep_features(train)
        test_prepped = prep_features(test)

        # Split train and test sets into features set and target set
        xtrain, ytrain = train_prepped.drop(columns=target), train_prepped[target]
        xtest, ytest = test_prepped.drop(columns=target), test_prepped[target]

        # Make sure that both train and test feature set have ALL features
        # and that they are in the same order
        xtrain, xtest = fill_missing_cols(xtrain, xtest)

        # Store in the final return dictionary with the split number
        # as the key
        train_test_dict[split_num] = {'train_dates':traindate_string, 
            'test_dates':testdate_string, 'xtrain':xtrain, 'ytrain':ytrain, 
            'xtest':xtest, 'ytest':ytest}
        i+=1

    return train_test_dict


def prep_features(df):
    '''
    Prepare features for a given dataset. Steps include:

    - Stripping whitespace from string columns
    - Imputing missing values with 0, "NONE", "UNKNOWN", or NaN
      values as appropriate
    - Aggregating and dummifying categorical variables
    - Using date variables to create features that indicate whether a
      given event occurred within a given time interval of an evaluation
    - Scaling numeric variables

    Inputs:
      df (pandas df): raw merged CSV produced by merge_and_collapse.py
        (combining all 6 input datasets of RCRA information)

    Returns: 
      cleaned df with all features generated, imputed, dummified,
        and scaled. Important to note that these operations are all run
        on an INDIVIDUAL data set AFTER a train/test split.
    '''
    # Strip whitespace from character cols
    df = strip_whitespace(df, ['EVALUATION_AGENCY', 'FOUND_VIOLATION',
        'FED_WASTE_GENERATOR', 'TRANSPORTER'])

    # Impute string columns with appropriate values as defined in global
    # IMPUTE_W_STR list
    for col in utils.IMPUTE_W_STR:
        df = impute_missing(df, col[0], col[1])

    # Impute 'avg_resolution_time' (by-facility average time from date
    # violation determined to date violation resolved) with the overall
    # full-universe median of single-facility average resolution time.
    # The intuition is that we wouldn't want to assume facilities with no
    # prior violations would have an average resolution time of 0, that
    # isn't logical. Assume that facilities without prior violations would
    # be "roughly average" in responding to a violation if they actually
    # do commit one.
    df.loc[:,'avg_resolution_time'] = df['avg_resolution_time'].astype(float)
    median_resolve_time = df['avg_resolution_time'].median()
    df = impute_missing(df, 'avg_resolution_time', median_resolve_time)

    # Aggregate NAICS code for industry to the highest level, enumerated
    # by the first two digits of the code. Source for industry codes:
    # https://www.naics.com/search-naics-codes-by-industry/
    df.loc[:,'naics'] = df['naics'].str.slice(0,2)

    # Aggregate RCRA violation codes to the highest level, enumerated by
    # the first three digits of the code. Source for violation codes:
    # https://echo.epa.gov/system/files/ndv_viol_type_0.pdf
    df.loc[:,'most_common_type'] = df['most_common_type'].str.slice(0,3)

    # Dummify categorical columns as defifned in global DUMMY_COLS list
    for col in utils.DUMMY_COLS:
        df = dummify_categorical(df, [col[0]], col[1])

    # Convert the string 'FOUND_VIOLATION' target variable to a numeric
    # indicator where 1 = Y or U (yes violation or unknown) and 0 = N (no viol)
    # IMPORTANT: we will treat 'U' (unknown) violations as 'flagged'
    # (committed violation) to be on the conservative side
    df['violation'] = np.where(df['FOUND_VIOLATION'] == 'N', 0, 1)

    # Create indicator columns for whether a facility falls within a
    # given enforcement or operating type, as defined in the
    # ENFORCEMENT_TYPES global list
    for letter in utils.ENFORCEMENT_TYPES:
        newcol = 'fenforce_' + letter
        df = create_indicator_for_contains(
            df, 'FULL_ENFORCEMENT', newcol, letter)
        newcol2 = 'op_tsdf_' + letter
        df = create_indicator_for_contains(
            df, 'OPERATING_TSDF', newcol2, letter)

    # Create indicator columns for whether a facility falls within
    # certain codes for active sites as defined in the ACTIVE_SITES
    # global list
    for letter in utils.ACTIVE_SITE:
        newcol = 'active_' + letter
        df = create_indicator_for_contains(
            df, 'ACTIVE_SITE', newcol, letter)

    # Create indicator columns for whether a facility falls within
    # certain reporting universes as laid out in the HREPORT_TYPES
    # global list
    for value, colname in zip(utils.HREPORT_TYPES, utils.HREPORT_COLS):
        df = create_indicator_for_contains(
            df, 'HREPORT_UNIVERSE_RECORD', colname, value)

    # Impute specified numeric columns as enumerated in the 
    # IMPUTE_W_0 global list
    for col in utils.IMPUTE_W_0:
        df = impute_missing(df, col, 0)

    # Create indicator variables to determine whether a violation,
    # vio flag, snc flag, or enforcement ocurred/was applied to a facility
    # within a 6-, 12-, 18-, 24-, 60-, and 120-month period of time
    # prior to a given evaluation. Important to note that this 'previous'
    # information is calculated using the PLANNING date (1/1) as the 
    # reference point rather than the EVALUATION date
    df = indicator_timeframe(
        df, 'months_since_viol', 'viol', 'plan_date', 'most_recent_viol')
    df['most_recent_vio_flag'] = pd.to_datetime(
        df['most_recent_vio_flag'], format='%Y-%m', errors='coerce')
    df = indicator_timeframe(
        df, 'months_since_vioflag', 'vioflag', 'plan_date', 'most_recent_vio_flag')
    df['most_recent_snc_flag'] = pd.to_datetime(
        df['most_recent_snc_flag'], format='%Y-%m', errors='coerce')
    df = indicator_timeframe(
        df, 'months_since_sncflag', 'sncflag', 'plan_date', 'most_recent_snc_flag')
    df['most_recent_enf'] = pd.to_datetime(
        df['most_recent_enf'], errors='coerce')
    df = indicator_timeframe(
        df, 'months_since_enf', 'enfflag', 'plan_date', 'most_recent_enf')

    # Scale numeric variables as enumerated in in SCALING_COLS global list
    for col in utils.SCALING_COLS:
        df = scaler(df, col)

    # Drop specific string/date columns that have been converted to features.
    # After this drop statement, only imputed and scaled variables of type
    # numeric or binary should remain.
    df.drop(columns=['FULL_ENFORCEMENT', 'HREPORT_UNIVERSE_RECORD',
        'FOUND_VIOLATION', 'eval_date', 'ACTIVE_SITE', 'OPERATING_TSDF',
        'most_recent_viol', 'most_recent_snc_flag', 'most_recent_vio_flag', 
        'viosnc_date', 'most_recent_enf', 'months_since_viol',
        'months_since_vioflag', 'months_since_sncflag', 
        'months_since_enf', 'plan_date'], inplace=True)

    return df

