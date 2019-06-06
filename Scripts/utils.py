'''
utilities for generate_features.py
'''
import numpy as np

# Force 'most_common_enf_type' to be read in as string, not int
INPUT_DTYPE = {'ID_NUMBER':str,'ACTIVITY_LOCATION_x':str,'EVALUATION_IDENTIFIER':str,
    'EVALUATION_TYPE':str, 'EVALUATION_DESC':str, 'EVALUATION_AGENCY':str,
    'EVALUATION_START_DATE':str, 'FOUND_VIOLATION':str, 'eval_date':str,
    'analysis_yearmonth':str, 'prev_violations':float, 'most_recent_viol':str,
    'most_common_type':str, 'avg_resolution_time':str, 'previous_vio_flags':float,
    'most_recent_vio_flag':str, 'previous_snc_flags':float, 'most_recent_snc_flag':str,
    'ACTIVITY_LOCATION_y':str, 'YRMONTH':str, 'VIO_FLAG':str, 'SNC_FLAG':str,
    'viosnc_date':str, 'yearmonth':str, 'current_vioflag':float, 'curent_sncflag':float,
    'FACILITY_NAME':str, 'ACTIVITY_LOCATION':str, 'FULL_ENFORCEMENT':str,
    'HREPORT_UNIVERSE_RECORD':str, 'STREET_ADDRESS':str, 'CITY_NAME':str,
    'STATE_CODE':str, 'ZIP_CODE':str, 'LATITUDE83':str, 'LONGITUDE83':str,
    'TRANSPORTER':str, 'ACTIVE_SITE':str, 'OPERATING_TSDF':str, 'naics':str,
    'year':str, 'analysis_year':str, 'previous_enfs':float, 'most_recent_enf':str,
    'most_commmon_enf_type':str, 'most_common_enf_agency':str, 'pmp_ct':float,
    'pmp_amt':float, 'fmp_ct':float, 'fmp_amt':float, 'fsc_ct':float, 'fsc_amt':float,
    'scr_ct':float, 'scr_amt':float}

# Identify extraneous columns to immediately drop from input CSV
DROPCOLS = ['ID_NUMBER','EVALUATION_IDENTIFIER','EVALUATION_DESC',
        'EVALUATION_START_DATE', 'analysis_yearmonth', 'ACTIVITY_LOCATION_y',
        'YRMONTH','VIO_FLAG','SNC_FLAG','yearmonth','FACILITY_NAME',
        'ACTIVITY_LOCATION','STREET_ADDRESS','CITY_NAME','STATE_CODE',
        'ZIP_CODE','LATITUDE83','LONGITUDE83','year','analysis_year']

# List of unique code types for enforcement_types and active_site
ENFORCEMENT_TYPES = ['L','I','B','S','T','H']
ACTIVE_SITE = ['H','P','A','C','S']

# List of hreport 'universes' to which a facility can belong,
# use these lists to create a series of binary indicator columns
HREPORT_TYPES = ['LQG', 'SQG', 'VSQG', 'TSDF', 'Operating TSDF', 
    'Other', 'Transporter']
HREPORT_COLS = ['h_lqg', 'h_sqg', 'h_vsqg', 'h_tsdf', 'h_op_tsdf', 
    'h_oth', 'h_transporter']

# List of tuples: first element is column to impute with a str or Nonetype
# value, second element is the value to impute
IMPUTE_W_STR = [('FED_WASTE_GENERATOR', 'UNKNOWN'), ('TRANSPORTER', 'UNKNOWN'),
    ('most_recent_viol', np.NaN), ('most_common_type', 'NONE'), 
    ('most_common_enf_type', 'NONE'), ('most_common_enf_agency', 'NONE')]

# List of tuples for columns to dummify: first element is column name,
# second element is the prefix to affix in the dummification process
DUMMY_COLS = [('ACTIVITY_LOCATION_x', 'state'), ('EVALUATION_TYPE', 'evtype'),
    ('TRANSPORTER', 'transporter'), ('EVALUATION_AGENCY', 'evagency'), 
    ('FED_WASTE_GENERATOR', 'fwg'), ('naics', 'naics'),
    ('most_common_type', 'most_common_type'), 
    ('most_common_enf_type', 'most_common_enf_type'),
    ('most_common_enf_agency', 'most_common_enf_agency')]

# List of numeric columns where to impute with value of 0
IMPUTE_W_0 = ['prev_violations', 'previous_vio_flags', 'current_vioflag', 
    'previous_snc_flags', 'current_sncflag', 'previous_enfs', 'pmp_ct', 
    'pmp_amt', 'fmp_ct', 'fmp_amt', 'fsc_ct', 'fsc_amt', 'scr_ct', 'scr_amt']

# List of numeric columns to scale using min-max scaling
SCALING_COLS = ['prev_violations','avg_resolution_time','previous_vio_flags',
    'previous_snc_flags', 'previous_enfs', 'pmp_ct','pmp_amt','fmp_ct','fmp_amt',
    'fsc_ct','fsc_amt','scr_ct','scr_amt']

# List of all features
features = ['prev_violations','avg_resolution_time','previous_vio_flags','previous_snc_flags',
'current_vioflag','current_sncflag','previous_enfs','pmp_ct','pmp_amt','fmp_ct','fmp_amt','fsc_ct',
'fsc_amt','scr_ct','scr_amt','state_AK','state_AL','state_AR','state_AS','state_AZ','state_CA',
'state_CO','state_CT','state_DC','state_DE','state_FL','state_GA','state_GU',
'state_HI','state_IA','state_ID','state_IL','state_IN','state_KS','state_KY','state_LA',
'state_MA','state_MD','state_ME','state_MI','state_MN','state_MO','state_MP','state_MS',
'state_MT','state_NC','state_ND','state_NE','state_NH','state_NJ','state_NM','state_NN',
'state_NV','state_NY','state_OH','state_OK','state_OR','state_PA','state_PR','state_RI',
'state_SC','state_SD','state_TN','state_TT','state_TX','state_UT','state_VA','state_VI',
'state_VT','state_WA','state_WI','state_WV','state_WY','evtype_CAC','evtype_CAV','evtype_CDI',
'evtype_CEI','evtype_CSE','evtype_FCI','evtype_FRR','evtype_FSD','evtype_FUI','evtype_GME',
'evtype_NIR','evtype_NRR','evtype_OAM','evtype_SNN','evtype_SNY','transporter_N',
'transporter_UNKNOWN','transporter_Y','evagency_B','evagency_C','evagency_E','evagency_L',
'evagency_N','evagency_S','evagency_T','evagency_X','fwg_1','fwg_2','fwg_3','fwg_N','fwg_UNKNOWN',
'naics_11','naics_21','naics_22','naics_23','naics_31','naics_32','naics_33','naics_42',
'naics_44','naics_45','naics_48','naics_49','naics_51','naics_52','naics_53','naics_54',
'naics_55','naics_56','naics_61','naics_62','naics_71','naics_72','naics_81','naics_92',
'most_common_type_261','most_common_type_262','most_common_type_263','most_common_type_264',
'most_common_type_265','most_common_type_266','most_common_type_268','most_common_type_270',
'most_common_type_271','most_common_type_273','most_common_type_279','most_common_type_FEA',
'most_common_type_FSS','most_common_type_NON','most_common_type_PCR','most_common_type_XXS',
'most_common_enf_type_80.0','most_common_enf_type_100.0','most_common_enf_type_110.0',
'most_common_enf_type_120.0','most_common_enf_type_130.0','most_common_enf_type_140.0',
'most_common_enf_type_150.0','most_common_enf_type_160.0','most_common_enf_type_170.0',
'most_common_enf_type_200.0','most_common_enf_type_210.0','most_common_enf_type_220.0',
'most_common_enf_type_230.0','most_common_enf_type_240.0','most_common_enf_type_250.0',
'most_common_enf_type_300.0','most_common_enf_type_310.0','most_common_enf_type_320.0',
'most_common_enf_type_330.0','most_common_enf_type_340.0','most_common_enf_type_380.0',
'most_common_enf_type_385.0','most_common_enf_type_410.0','most_common_enf_type_420.0',
'most_common_enf_type_510.0','most_common_enf_type_520.0','most_common_enf_type_590.0',
'most_common_enf_type_610.0','most_common_enf_type_620.0','most_common_enf_type_710.0',
'most_common_enf_type_730.0','most_common_enf_type_800.0','most_common_enf_type_810.0',
'most_common_enf_type_820.0','most_common_enf_type_830.0','most_common_enf_type_860.0',
'most_common_enf_type_NONE','most_common_enf_agency_E','most_common_enf_agency_NONE',
'most_common_enf_agency_S','fenforce_L','op_tsdf_L','fenforce_I','op_tsdf_I','fenforce_B',
'op_tsdf_B','fenforce_S','op_tsdf_S','fenforce_T','op_tsdf_T','fenforce_H','op_tsdf_H',
'active_H','active_P','active_A','active_C','active_S','h_lqg','h_sqg','h_vsqg','h_tsdf',
'h_op_tsdf','h_oth','h_transporter','viol_6mo','viol_12mo','viol_18mo','viol_24mo',
'viol_60mo','viol_120mo','vioflag_6mo','vioflag_12mo','vioflag_18mo','vioflag_24mo','vioflag_60mo',
'vioflag_120mo','sncflag_6mo','sncflag_12mo','sncflag_18mo','sncflag_24mo','sncflag_60mo',
'sncflag_120mo','enfflag_6mo','enfflag_12mo','enfflag_18mo','enfflag_24mo','enfflag_60mo',
'enfflag_120mo']
