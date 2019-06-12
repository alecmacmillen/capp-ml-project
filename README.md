# capp-ml-project

CAPP30254 ML project for Tim Hannifan, Alec MacMillen, Quinn Underriner

## **FOLDER STRUCTURE**

### **Data**

This folder contains all raw input, intermediate built, and final output data (mostly CSVs). Contains 4 subfolders:

1. **Input**: Contains input data from EPA's RCRA data download resource. The raw files are titled RCRA_ENFORCEMENTS.csv, RCRA_EVALUATIONS.csv, RCRA_FACILITIES.csv, RCRA_NAICS.csv, RCRA_VIOLATIONS.csv, and RCRA_VIOSNC_HISTORY.csv. There is a **Small** subfolder that contains sample subsets of each dataset for testing purposes. Original raw datasets have been zipped to save space.
2. **Built**: Contains the file *updated_full_built.zip*, a zipped CSV of the output data produced by running *Scripts/merge_and_collapse.py* on the input datasets. This is the first step that combines all 6 RCRA datasets into a master dataset with all information contained.
3. **Documentation**: Contains an Excel file *all_features.xlsx* that lists all features, a Word document with feature brainstorming, and a PDF to NAICS industry code documentation.
4. **Results**: Contains all final output results. The files titled *split_n_models.csv* contains all the trained models and their performance metrics for splits 0-15 (as laid out in *Scripts/model_specs.py*). The rest of the files all correspond to the single best-performing model for the longest train-test split (split #15), with training years 2000-2015 and testing years 2017.
   - *predicted_violations_2017.csv* is the full list of predicted and actual labels for the 2017 testing set using the best-performing model trained on the 2000-2015 training set.
   - *targeted_facility_list_2017.csv* contains the top 10% (for 10% of precision) evaluations labelled as likely violations in 2017 - this is the final deliverable of facilities selected for inspection/extra scrutiny.
   - *feature_importances.csv* lists the features and their relative weight by importance according to our best-performing model.
   - *yearly_precision_graph.png* is a graph that shows the best-performing model of each type's precision at 10% for the testing period of all 16 splits (numbers 0-15), which aided in our selection of the best-performing model.
   - *precision_recall_curve.png* is a graph that plots precision and recall at various thresholds for our best-performing model according to precision at 10% for split number 15, which was a random forest classifier using Gini splitting criteria, 10 estimators, and no maximum depth.

### **Report**

This folder contains copies of our original proposal, progress check-in and final report.

### **Scripts**

This folder contains all programs used to run the project. In order of use:

1. *merge_and_collapse.py* - this is the first program that's run on the 6 raw input CSVs. Essentially, it reads in all 6 files and merges the evaluations, facilities, and NAICS information together before calculating summary statistics by facility using ONLY information from violations, enforcements, and viosnc-history that occurred *before* a given evaluation date. The end result is a merged output CSV that has one row for every row found in the original RCRA_EVALUATIONS data. (Essentially, we're taking a bunch of new information and adding it to evaluations so that we can predict an evaluation's outcome.) These aggregations occur at the year level. The result of this program is *updated_full_built.zip* in the **Data/Built** folder.
2. *generate_features.py* - this program is called from within *run_models.py* to dynamically generate features for each individual train/test set. It performs a series of imputations, dummifications, conversions, and scalings which are documented in the code comments. The *clean_and_split()* function takes the master built dataset *updated_full_built.csv* and iteratively creates train/test split using the *prep_features()* wrapping function. The end result is a dictionary where the keys are serial integers (the split number) and the values are the dictionaries corresponding to that split: they contain train/test metadata, as well as the x-train, y-train, x-test, and y-test datasets, which are all then passed to *run_models.py*. This module does not get called on its own, it's called from within *run_models.py*. This module also calls in *utils.py*, which contains some hard-coded utility values for convenience.
3. *run_models.py* - this module actually runs the train-test split generation function from *generate_features.py* and then calls the magic loop *run_all_models()* function from *pipeline.py*, which iteratively trains and tests all models and specifications outlined in *model_specs.py*. All results are output to a CSV file that contains all model information and performance for that split.
4. *pipeline.py* - pipeline created for the class which contains magic loop, metric calculations, etc. that are all called by *run_models.py*.
5. *utils.py* - contains some hard-coded values related to data types and feature generation that is called into other programs.
6. *model_specs.py* - contains model metadata that is passed to *run_models.py* so that different models with different parameters are trained and evaluated.
7. *visualizations.py* - general visualization functions for exploratory data analysis.
8. *pick_best_model.py* - reads in the by-split model performance CSVs created by *run_models.py* and analyzes the best-performing model of each type for each split, then plots performance over time. This program outputs *yearly_precision_graph.png* in the **Data/Results** folder.
9. *output_list.py* - uses the best-performing model determined by *pick_best_model.py* to generate final list of predicted class labels, targeted facilities for inspection, precision-recall curve, and feature importances. This code create the *predicted_violations_2017.csv*, *targeted_facility_list_2017.csv*, *precision_recall_curve.png*, and *feature_importances.csv* files in the **Data/Results** folder.