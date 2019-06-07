'''
merge_and_collapse

Combine EPA RCRA data, calculate summary statistics from RCRA_VIOLATIONS,
RCRA_ENFORCEMENTS, and RCRA_VIOSNC_HIST files specifically for the time
at which evaluations were conducted and merge.
'''

import sys
import math
import pandas as pd
import numpy as np
from functools import reduce
pd.options.mode.chained_assignment = None # default = 'warn'


def read_and_convert_dates(file, old_datecols, new_datecols, dateformat='%m/%d/%Y'):
    '''
    Read in RCRA data and automatically convert strings in date columns to datetime objects.

    Inputs:
      file (str): filepath location for RCRA data
      old_datecols (list of str): names of date columns in input file
      new_datecols (list of str): corresponding new names for date columns in pandas df
      dateformat (str): string format of date column in input file, default %m/%d/%Y (01/01/1980)

    Returns:
      loaded and date-converted pandas df
    '''
    df = pd.read_csv(file)
    for old_datecol, new_datecol in zip(old_datecols, new_datecols):
        df[new_datecol] = pd.to_datetime(df[old_datecol], format=dateformat, errors="coerce")
    return df


def filter_viol_on_year(violations_df, year):
    '''
    For a given year, calculate features for a facility as a "snapshot
    in time", that is, only using information available in the violations dataset
    *up until that year* (so no future information is used). For a given
    year pair, calculate the total number of previous violations, date of
    most recent violation, most common violation type, and average time to resolve.
    Then merge summary statistics together on facility ID and return as a dataframe.

    Inputs:
      violations_df (pandas dataframe): information on violations
      year (period type of 'Y'): yearmonth to filter on (all information BEFORE
        this year is captured)

    Returns:
      final_frame (pandas dataframe): summary statistics by facility for the period
        from the beginning of the data to the year specified
    '''
    # Filter the dataframe to include only observations before yrmonth
    # and create a 'time_to_resolve' variable based on the violation determined
    # and resolution date
    outdf = violations_df[violations_df['year'] < year]
    gb_obj = outdf.groupby(['ID_NUMBER'])
    
    # Calculate total count of previous violations
    total_prev = get_counts(gb_obj, 'viol_date', 'prev_violations')

    # Find date of most recent violation
    most_recent = get_recent(gb_obj, 'viol_date', 'most_recent_viol')
    
    # Calculate most common previous violation type
    most_common_type = most_common(gb_obj, 'VIOLATION_TYPE','most_common_type')

    # Calculate average time to resolution for previous violations
    avg_resolution_time = outdf.groupby(['ID_NUMBER'])['time_to_resolve'].mean()
    avg_resolution_time = avg_resolution_time.to_frame().reset_index()
    avg_resolution_time.rename(columns={'time_to_resolve':'avg_resolution_time'}, inplace=True)

    # Merge summary statistics together into one final frame
    df_list = [total_prev, most_recent, most_common_type, avg_resolution_time]
    final_frame = reduce(lambda left, right: pd.merge(left, right, on=['ID_NUMBER'],
        how='inner'), df_list)

    final_frame.loc[:, 'year'] = year

    return final_frame


def yearly_viol_info(violations_df, startyear, endyear):
    '''
    Wrapper function for filter_viol_on_year Takes cleaned violations dataframe
    and produces stacked dataframe of all information on previous violations by facility
    and year from 1984 to 2018.

    Inputs:
      violations_df (pandas dataframe): cleaned violations dataframe

    Returns:
      yearly_summary (pandas dataframe): past violations summary by facility and
        date in year format.
    '''
    # Generate dataframe for each year with summary stats
    yearlist = list(range(startyear, endyear+1))
    dfs_dict = {}
    i = 1
    print("Producing violations summary")
    for year in yearlist:
        print("Progress: year ", i, " of ", len(yearlist))
        dfs_dict[year] = filter_viol_on_year(violations_df, year)
        i += 1

    # Concatenate all dfs and return
    yearly_summary = pd.concat(dfs_dict.values(), sort=True)
    return yearly_summary



def filter_enf_on_year(enforcements_df, year):
    '''
    For a given year-month pair, calculate features for a facility as a "snapshot
    in time", that is, only using information available in the enforcements dataset
    *up until that year-month* (so no future information is used). For a given
    year-month pair, calculate the total number of previous enforcements, date of
    most recent violation, most common enforcement type and agency, and count and
    total amounts of monetary enforcement penalties. Then merge summary statistics 
    together on facility ID and return as a dataframe.

    Inputs:
      enforcements_df (pandas dataframe): information on enforcements
      year (int): year to filter on (all information BEFORE
        this year is captured)

    Returns:
      final_frame (pandas dataframe): summary statistics by facility for the period
        from the beginning of the data to the yrmonth specified
    '''
    # Filter the dataframe to include only observations before yrmonth
    # and create a groupby object to use to calculate summary stats
    outdf = enforcements_df[enforcements_df['year'] < year]
    gb_obj = outdf.groupby(['ID_NUMBER'])

    # Calculate counts, date of most recent, and count/amount of previous
    # financial enforcement actions
    enf_prev = get_counts(gb_obj, 'enf_date', 'previous_enfs')
    enf_recent = get_recent(gb_obj, 'enf_date', 'most_recent_enf')
    most_common_enf_type = most_common(gb_obj, 'type_category', 'most_common_enf_type')
    most_common_enf_agency = most_common(gb_obj, 'ENFORCEMENT_AGENCY', 'most_common_enf_agency')
    pmp_ct = get_counts(gb_obj, 'PMP_AMOUNT', 'pmp_ct')
    pmp_amt = get_sum(gb_obj, 'PMP_AMOUNT', 'pmp_amt')
    fmp_ct = get_counts(gb_obj, 'FMP_AMOUNT', 'fmp_ct')
    fmp_amt = get_sum(gb_obj, 'FMP_AMOUNT', 'fmp_amt')
    fsc_ct = get_counts(gb_obj, 'FSC_AMOUNT', 'fsc_ct')
    fsc_amt = get_sum(gb_obj, 'FSC_AMOUNT', 'fsc_amt')
    scr_ct = get_counts(gb_obj, 'SCR_AMOUNT', 'scr_ct')
    scr_amt = get_sum(gb_obj, 'SCR_AMOUNT', 'scr_amt')

    # Merge all together into a single df for the year
    df_list = [enf_prev, enf_recent, most_common_enf_type, most_common_enf_agency,
        pmp_ct, pmp_amt, fmp_ct, fmp_amt, fsc_ct, fsc_amt, scr_ct, scr_amt]
    final_frame = reduce(lambda left, right: pd.merge(left, right, on=['ID_NUMBER'],
        how='inner'), df_list)

    final_frame.loc[:, 'year'] = year

    return final_frame


def yearly_enf_info(enforcements_df, startyear, endyear):
    '''
    Wrapper function for filter_enf_on_year. Takes cleaned enforcements dataframe
    and produces stacked dataframe of all information on previous enforcements on facility
    and year from 1984 to 2018.

    Inputs:
      enforcements_df (pandas dataframe): cleaned enforcements dataframe

    Returns:
      yearly_summary (pandas dataframe): past enforcements summary by facility and
        date in year format.
    '''
    # Generate dataframe for each year with summary stats
    yearlist = list(range(startyear, endyear+1))

    # Generate dataframe for each year with summary stats
    dfs_dict = {}
    i = 1
    print("Producing enforcements summary")
    for year in yearlist:
        print("Progress: year ", i, " of ", len(yearlist))
        dfs_dict[year] = filter_enf_on_year(enforcements_df, year)
        i += 1

    # Concatenate all dfs and return
    yearly_summary = pd.concat(dfs_dict.values(), sort=True)
    return yearly_summary


def filter_viosnc_on_year(viosnc_df, year):
    '''
    For a given year pair, calculate features for a facility as a "snapshot
    in time", that is, only using information available in the viosnc dataset
    *up until that year* (so no future information is used). For a given
    year, calculate the total number of previous flags of each type, date of
    most recent flag, and whether facility is currently under a flag of either type. 
    Then merge summary statistics together on facility ID and return as a dataframe.

    Inputs:
      viosnc_df (pandas dataframe): information on viosnc
      year (int): year to filter on (all information BEFORE
        this year is captured)

    Returns:
      final_frame (pandas dataframe): summary statistics by facility for the period
        from the beginning of the data to the yrmonth specified
    '''
    # Calculate summary stats for "VIO" flag
    outdf = viosnc_df[(viosnc_df['year'] < year) & (viosnc_df['VIO_FLAG']=='Y')]
    gb_obj = outdf.groupby(['ID_NUMBER'])
    vio_prev = get_counts(gb_obj, 'VIO_FLAG', 'previous_vio_flags')
    vio_recent = get_recent(gb_obj, 'year', 'most_recent_vio_flag')

    # Calculate summary stats for "SNC" flag
    outdf2 = viosnc_df[(viosnc_df['year'] < year) & (viosnc_df['SNC_FLAG']=='Y')]
    gb_obj2 = outdf2.groupby(['ID_NUMBER'])
    snc_prev = get_counts(gb_obj2, 'SNC_FLAG', 'previous_snc_flags')
    snc_recent = get_recent(gb_obj2, 'year', 'most_recent_snc_flag')

    # Calculate whether the facility is currently under a flag as of current yearmonth
    latest_idx = outdf.groupby(['ID_NUMBER'])['year'].transform(max) == outdf['year']
    currently_under = outdf[latest_idx]
    currently_under['current_vioflag'] = np.where(currently_under['VIO_FLAG'] == 'Y', 1, 0)
    currently_under['current_sncflag'] = np.where(currently_under['SNC_FLAG'] == 'Y', 1, 0)
    currently_under.drop(
        columns=['ACTIVITY_LOCATION', 'YRMONTH', 'VIO_FLAG', 'SNC_FLAG', 'year']) 

    # Merge all into final frame
    df_list = [vio_prev, vio_recent, snc_prev, snc_recent, currently_under]
    final_frame = reduce(lambda left, right: pd.merge(left, right, on=['ID_NUMBER'],
        how='inner'), df_list)

    final_frame.loc[:, 'year'] = year

    return final_frame


def yearly_viosnc_info(viosnc_df, startyear, endyear):
    '''
    Wrapper function for filter_viosnc_on_year. Takes cleaned viosnc dataframe
    and produces stacked dataframe of all information on previous viosnc on facility
    and year from 1984 to 2018.

    Inputs:
      viosnc_df (pandas dataframe): cleaned viosnc dataframe

    Returns:
      yearly_summary (pandas dataframe): past viosnc flags summary by facility and
        date in year format.
    '''
    # Create list of years over which to iterate
    yearlist = list(range(startyear, endyear+1))

    # Generate dataframe for each yearmonth with summary stats
    dfs_dict = {}
    i = 1
    print("Producing viosnc summaries")
    for year in yearlist:
        print("Progress: year ", i, " of ", len(yearlist))
        dfs_dict[year] = filter_viosnc_on_year(viosnc_df, year)
        i += 1

    # Concatenate all dfs and return
    yearly_summary = pd.concat(dfs_dict.values(), sort=True)
    return yearly_summary


def get_counts(gb_obj, oldcol, newcol):
    '''
    Helper function for calculating summary statistic (count) of a given
    column after grouping-by on ID_NUMBER to create gb_obj.

    Inputs:
      gb_obj (pandas groupby object): pandas df pre-filtered on yearmonth
        and pre-grouped by ID_NUMBER
      oldcol (str): name of column in the original dataframe to summarize
      newcol (str): name to rename old column to

    Returns:
      pandas df of summary stat by facility ID_NUMBER
    '''
    prev_count = gb_obj[oldcol].count()
    prev_count = prev_count.to_frame().reset_index()
    prev_count.rename(columns={oldcol:newcol}, inplace=True)
    return prev_count


def get_recent(gb_obj, oldcol, newcol):
    '''
    Helper function for calculating summary statistic (most recent) of a given
    column after grouping-by on ID_NUMBER to create gb_obj.

    Inputs:
      gb_obj (pandas groupby object): pandas df pre-filtered on yearmonth
        and pre-grouped by ID_NUMBER
      oldcol (str): name of column in the original dataframe to summarize
      newcol (str): name to rename old column to

    Returns:
      pandas df of summary stat by facility ID_NUMBER
    '''
    most_recent = gb_obj[oldcol].max()
    most_recent = most_recent.to_frame().reset_index()
    most_recent.rename(columns={oldcol:newcol}, inplace=True)
    return most_recent


def most_common(gb_obj, oldcol, newcol):
    '''
    Helper function for calculating summary statistic (mode) of a given
    column after grouping-by on ID_NUMBER to create gb_obj.

    Inputs:
      gb_obj (pandas groupby object): pandas df pre-filtered on yearmonth
        and pre-grouped by ID_NUMBER
      oldcol (str): name of column in the original dataframe to summarize
      newcol (str): name to rename old column to

    Returns:
      pandas df of summary stat by facility ID_NUMBER
    '''
    most_common = gb_obj[oldcol].agg(lambda x: pd.Series.mode(x)[0])
    most_common = most_common.to_frame().reset_index()
    most_common.rename(columns={oldcol:newcol}, inplace=True)
    return most_common


def get_sum(gb_obj, oldcol, newcol):
    '''
    Helper function for calculating summary statistic (sum) of a given
    column after grouping-by on ID_NUMBER to create gb_obj.

    Inputs:
      gb_obj (pandas groupby object): pandas df pre-filtered on yearmonth
        and pre-grouped by ID_NUMBER
      oldcol (str): name of column in the original dataframe to summarize
      newcol (str): name to rename old column to

    Returns:
      pandas df of summary stat by facility ID_NUMBER
    '''
    sum_amt = gb_obj[oldcol].sum()
    sum_amt = sum_amt.to_frame().reset_index()
    sum_amt.rename(columns={oldcol:newcol}, inplace=True)
    return sum_amt


def go(args):
    '''
    Perform ALL load, clean, summarize, and merge operations to produce one
    final dataset that combines information from all 6 input EPA datasets:
      1. RCRA_EVALUATIONS
      2. RCRA_FACILILTIES
      3. RCRA_NAICS
      4. RCRA_VIOLATIONS
      5. RCRA_VIOSNC_HIST
      6. RCRA_ENFORCEMENTS

    The goal of this program is to produce a final dataset that preserves 
    all the rows from RCRA_EVALUATIONS (which contains information about 
    individual evaluations and whether or not they resulted in a found
    violation) and combine it with historical information from RCRA_VIOLATIONS,
    RCRA_VIOSONC_HIST, and RCRA_ENFORCEMENTS while *only* using information
    from *before* the given evaluation date (so that we're not using future
    information to predict current outcomes).

    Inputs:
      args (list of filenames): see usage

    Returns: final merged dataframe that is written out to the specified
      outfile_path.
    '''
    usage = ("usage: python3 merge_and_collapse.py <evaluations-file> <facilities-file>"
             " <industry-file> <violations-file> <viosnc-file> <enforcements-file>"
             " <outfile_path>")
    if len(args) != 8:
        print(usage)
        sys.exit(1)

    # RCRA_EVALUATIONS: read in and merge with RCRA_NAICS and RCRA_FACILITIES
    # to get by-facility information for each evaluation
    evaluations = read_and_convert_dates(args[1],
        ['EVALUATION_START_DATE'], ['eval_date'])
    evaluations['year'] = evaluations['eval_date'].dt.year
    evaluations['plan_date'] = pd.to_datetime('01/01/' + evaluations['year'].astype(str),
        format = '%m/%d/%Y')
    evaluations
    facilities = pd.read_csv(args[2])

    # Some facilities have more than one classification, so we'll use the
    # industry classifier that appears most often
    industries = pd.read_csv(args[3])
    industry_gb = industries.groupby(['ID_NUMBER'])
    industry_mode = most_common(industry_gb, 'NAICS_CODE', 'naics')


    merged = evaluations.merge(facilities, on='ID_NUMBER', how='left')
    merged = merged.merge(industry_mode, on='ID_NUMBER', how='left')


    # RCRA_VIOLATIONS: read in, clean, then get by-monthly summary statistics
    # with which to merge onto the final df
    violations = read_and_convert_dates(args[4],
        ['DATE_VIOLATION_DETERMINED','ACTUAL_RTC_DATE','SCHEDULED_COMPLIANCE_DATE'], 
        ['viol_date','rtc_date','compliance_date'])
    # Create yearmonth variable for filtering
    violations['year'] = violations['viol_date'].dt.year
    # Drop a few outlier observations that occur before 1980
    violations = violations[violations['year'] >= 1980]
    # Calculate 'time_to_resolve' violation as difference between violation determined date
    # and actual compliance date
    violations.loc[:, 'time_to_resolve'] = (violations['rtc_date'] - violations['viol_date']).dt.days
    # Calculate by-monthly statistics on violations history
    violations_summary = yearly_viol_info(violations, 1984, 2019)
    # Merge onto evaluations/facilities/industry dataset 
    merged = merged.merge(violations_summary, on=['ID_NUMBER','year'], how='left')


    # RCRA_VIOSNC_HISTORY: read in, clean, then get by-monthly summary statistics
    # with which to merge onto the final df
    viosnc = read_and_convert_dates(args[5],
        ['YRMONTH'], ['viosnc_date'], '%Y%m')
    # Drop a few outlier observations that occur before 1980 and convert the 'YRMONTH' 
    # variable to 'yearmonth' for summarizing
    viosnc['year'] = viosnc['viosnc_date'].dt.year
    viosnc = viosnc[viosnc['year'] >= 1980]
    # Calculate by-monthly statistics on viosnc flag history
    viosnc_summary = yearly_viosnc_info(viosnc, 1984, 2019)
    # Merge onto evaluations/faciltiies/industries/violations dataset
    merged = merged.merge(viosnc_summary, on=['ID_NUMBER', 'year'], how='left')


    # RCRA_ENFORCEMENTS: read in, clean, thhen get by-monthly summary statistics 
    # with which to merge onto the final df. For enforcements, we will aggregate by year,
    # which provides a slightly lower level of granularity/detail, but has the advantage
    # of not creating an intractably large dataframe (25m+ obs) which makes merging difficult.
    enforcements = read_and_convert_dates(args[6],
        ['ENFORCEMENT_ACTION_DATE'], ['enf_date'])
    # Create year variable for filtering
    enforcements.loc[:,'year'] = pd.to_datetime(enforcements['enf_date']).dt.year
    # Drop obs before 1980
    enforcements = enforcements[enforcements['year'] >= 1980]
    
    # Here we're making the assumption that enforcement types end in 0 
    # unless they're one of type 385, 425, or 865 as defined in the nationally
    # recognized enforcement types list found at the following website link:
    # https://echo.epa.gov/system/files/ndv_enf_type.pdf.
    enforcements['type_category'] = enforcements['ENFORCEMENT_TYPE'].str.slice(2,)
    type_mask = ~enforcements['type_category'].isin(['385','425','865'])
    enforcements['type_category'].loc[type_mask] = enforcements['type_category'].str.slice(0,2) + '0'
    # Calculate by-monthly statistics on enforcement history
    enforcement_summary = yearly_enf_info(enforcements, 1984, 2019)
    merged = merged.merge(enforcement_summary, on=['ID_NUMBER', 'year'], how='left')

    merged.to_csv(args[7], header=True, index=False)



if __name__ == "__main__":
    go(sys.argv)


