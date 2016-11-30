# possible multilabel classifier: SVM, Nearest Neighbours, Decision Trees, Random Forest
# Explore scikit-multilearn (includes Meka wrapper)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
import os
import time
from datetime import datetime as dt
import sys
import argparse
from Occupancy import dred_occupancy as do
from NILMTK.angga import nilmtkDREDappliance as nil
from RLO import RLO as rlo
from imblearn.combine import SMOTEENN

def perf_measure_room(test_ground_truth, test_prediction):
  TP, FP, TN, FN, total_sample, precision, recall, F = 0, 0, 0, 0, 0, 0, 0, 0;
  num_predictions = test_prediction.shape[0];
  num_rooms = test_prediction.shape[1];  
  for i in range(num_predictions):
    for j in range(num_rooms):
      if test_ground_truth[i][j]==test_prediction[i][j]==1: TP += 1;
      elif test_ground_truth[i][j]==0 and test_prediction[i][j]==1: FP += 1;
      elif test_ground_truth[i][j]==test_prediction[i][j]==0: TN += 1;
      elif test_ground_truth[i][j]==1 and test_prediction[i][j]==0: FN += 1;
  try:
    precision = TP / ((FP + TP)*1.0);
  except:
    precision = 0.0;
  try:
    recall = TP / ((FN + TP)*1.0);
  except:
    recall = 0.0;
  try:
    F = 2*precision*recall/((precision+recall)*1.0);
  except:
    F = 0.0;
  total_sample = TP + FP + TN + FN;
  TP = TP / (total_sample*1.0);
  FP = FP / (total_sample*1.0);	
  TN = TN / (total_sample*1.0);	
  FN = FN / (total_sample*1.0);
  # print("TP %s, FP %s, TN %s, FN %s, precision %s, recall %s, F %s" % TP, FP, TN, FN, precision, recall, F);
  return TP, FP, TN, FN, precision, recall, F;

def perf_measure_people(test_ground_truth, test_prediction):
  TP, FP, TN, FN, total_sample, precision, recall, F = 0, 0, 0, 0, 0, 0, 0, 0;
  num_predictions = test_prediction.shape[0];
  for i in range(num_predictions):
      if test_ground_truth[i]==test_prediction[i]==1: 
        TP += test_prediction[i];
      elif test_ground_truth[i] < test_prediction[i]: 
        TP += test_prediction[i];
        FP += test_prediction[i] - test_ground_truth[i];
      elif test_ground_truth[i]==test_prediction[i]==0: 
        TN += test_prediction[i];
      elif test_ground_truth[i] > test_prediction[i]: 
        TN += test_prediction[i];
        FN += test_prediction[i] - test_ground_truth[i];
  try:
    precision = TP / ((FP + TP)*1.0);
  except:
    precision = 0.0;
  try:
    recall = TP / ((FN + TP)*1.0);
  except:
    recall = 0.0;
  try:
    F = 2*precision*recall/((precision+recall)*1.0);
  except:
    F = 0.0;
  total_sample = TP + FP + TN + FN;
  TP = TP / (total_sample*1.0);
  FP = FP / (total_sample*1.0);	
  TN = TN / (total_sample*1.0);	
  FN = FN / (total_sample*1.0);
  return TP, FP, TN, FN, precision, recall, F;

# merge smart meter max and avg, appliances powers, house level occupancy, and appliance group using predictive methods
def merge_features():
#  train_features = pd.concat([sm_train, group_train], axis=1);
  # normalize
  norm_ap_train = appliance_power_train.drop('isempty', axis=1);
  min_max_scaler = preprocessing.MinMaxScaler();
  norm_ap_train = min_max_scaler.fit_transform(norm_ap_train.values); # sometimes contains NaN
  norm_ap_train = pd.DataFrame(norm_ap_train);
  norm_ap_test = appliance_power_test.drop('isempty', axis=1);
  min_max_scaler = preprocessing.MinMaxScaler();
  norm_ap_test = min_max_scaler.fit_transform(norm_ap_test.values); # sometimes contains NaN
  norm_ap_test = pd.DataFrame(norm_ap_test);
  train_features = sm_train;
  train_features[0]  = norm_ap_train.ix[:,0];
  train_features[1]  = norm_ap_train.ix[:,1];
  train_features[2]  = norm_ap_train.ix[:,2];
  train_features[3]  = norm_ap_train.ix[:,3];
  train_features[4]  = norm_ap_train.ix[:,4];
  train_features[5]  = norm_ap_train.ix[:,5];
  train_features[6]  = norm_ap_train.ix[:,6];
  train_features[7]  = norm_ap_train.ix[:,7];
  train_features[8]  = norm_ap_train.ix[:,8];
  train_features[9]  = norm_ap_train.ix[:,9];
  train_features[10] = norm_ap_train.ix[:,10];
  train_features[11] = norm_ap_train.ix[:,11];
#  test_features = pd.concat([sm_test, group_test], axis=1);
  test_features = sm_test;
  test_features[0]  = norm_ap_test.ix[:,0];
  test_features[1]  = norm_ap_test.ix[:,1];
  test_features[2]  = norm_ap_test.ix[:,2];
  test_features[3]  = norm_ap_test.ix[:,3];
  test_features[4]  = norm_ap_test.ix[:,4];
  test_features[5]  = norm_ap_test.ix[:,5];
  test_features[6]  = norm_ap_test.ix[:,6];
  test_features[7]  = norm_ap_test.ix[:,7];
  test_features[8]  = norm_ap_test.ix[:,8];
  test_features[9]  = norm_ap_test.ix[:,9];
  test_features[10] = norm_ap_test.ix[:,10];
  test_features[11] = norm_ap_test.ix[:,11];
  return train_features, test_features;

####################  
# START OF PROGRAM #
####################

sampling_rate = 1; # in seconds
feature_length = 60; # in seconds, defaults to 15 minutes
classifier = 0;
dataset_loc = '../dataset/DRED.h5';

# training set is fairly distributed
train_start = "2015-07-05";
train_end = "2015-08-30";
train_end_nil = "2015-08-31";

test_start = "2015-09-01";
test_end = "2015-09-08";
test_end_nil = "2015-09-09";

parser = argparse.ArgumentParser();
parser.add_argument("--cl", help="Classifier, 0=SVM RBF kernel, etc");
parser.add_argument("--sr", help="Sampling rate");
parser.add_argument("--fl", help="Feature length");
parser.add_argument("--str", help="Start of train, format is YYYY-MM-DD");
parser.add_argument("--etr", help="End of train, format is YYYY-MM-DD");
parser.add_argument("--ste", help="Start of test, format is YYYY-MM-DD");
parser.add_argument("--ete", help="End of test, format is YYYY-MM-DD");
args = parser.parse_args();

if args.cl:
  classifier = int(args.cl); # in seconds 

if args.sr:
  sampling_rate = int(args.sr); # in seconds

if args.fl:
  feature_length = int(args.fl); # in seconds 

if args.str:
  train_start = args.str;

if args.etr:
  train_end = args.etr;

if args.ste:
  test_start = args.ste;

if args.ete:
  test_end = args.ete;

if (feature_length % sampling_rate) > 1:
  print ("feature length must be divisible by, minimum twice, sampling_rate. exiting program...");
  sys.exit();

# Extract features and ground truth for both training and testing
# compute room level occupancy ground truth
#train_gt_room = do.ro_gt(train_start, train_end, feature_length);
#test_gt_room = do.ro_gt(test_start, test_end, feature_length);

# compute disaggregated power per appliances
appliance_power_test, appliance_power_test_gt, co_model, appliance_power_train = nil.nilmtkDREDfunc(dataset_loc, train_start, train_end_nil, test_start, test_end_nil, feature_length);
# compute room level occupancy ground truth and compute which appliances are on or off based on grouping rules
train_gt_room, group_train = rlo.occ_group_generator(dataset_loc, train_start, train_end_nil, feature_length, co_model);
test_gt_room, group_test = rlo.occ_group_generator(dataset_loc, test_start, test_end_nil, feature_length, co_model);
# compute aggregated smart meter features
sm_train = do.get_smf(train_start, train_end, feature_length);
sm_test = do.get_smf(test_start, test_end, feature_length);

# remove timezone info from room level occupancy dataframe
train_gt_room.index.tz = None;
train_gt_room = train_gt_room.shift(periods=2, freq='H');
test_gt_room.index.tz = None;
test_gt_room = test_gt_room.shift(periods=2, freq='H');

## Only pick index which has room level occupancy ground truth
sm_train = sm_train.loc[train_gt_room.index];
sm_test = sm_test.loc[test_gt_room.index];

# remove timezone info from aggregated power dataframe
appliance_power_train.index.tz = None;
appliance_power_train = appliance_power_train.shift(periods=2, freq='H'); 
appliance_power_train = appliance_power_train.loc[train_gt_room.index];
appliance_power_test.index.tz = None;
appliance_power_test = appliance_power_test.shift(periods=2, freq='H');
appliance_power_test = appliance_power_test.loc[test_gt_room.index];

# remove timezone info from grouped appliances dataframe
group_train.index.tz = None;
group_train = group_train.shift(periods=2, freq='H');
group_train = group_train.loc[train_gt_room.index];
group_test.index.tz = None;
group_test = group_test.shift(periods=2, freq='H');
group_test = group_test.loc[test_gt_room.index];

train_features, test_features = merge_features();

# drop NAs
train_features = train_features.dropna();
test_features = test_features.dropna();
train_gt_room = train_gt_room.loc[train_features.dropna().index];
test_gt_room = test_gt_room.loc[test_features.dropna().index];

# resample for each room
resampled_train_features = [];
resampled_train_gt_room = [];

sm = SMOTEENN(); 

for i in range(0, len(train_gt_room.columns)):
  X_resampled, y_resampled = sm.fit_sample(train_features, train_gt_room.ix[:,i]);
  # resampled_train_features.append(X_resampled);
  # resampled_train_gt_room.append(y_resampled);
  
  # classif_room = OneVsRestClassifier(svm.SVC(kernel='rbf'));
  classif_room = ensemble.RandomForestClassifier();
  classif_room.fit(X_resampled.astype(float), y_resampled.astype(int));
  test_prediction_room = classif_room.predict(test_features);
  precision_r, recall_r, F_r, supp_r = precision_recall_fscore_support(test_gt_room.ix[:,i].astype(int), test_prediction_room, average='micro');
  result_r = "room " + str(i) + ": " +  str(precision_r) + "," + str(recall_r) + "," + str(F_r);
  print result_r;
  with open('Results' + os.path.sep + 'result_roses_dred.csv', "a") as myfile:
    myfile.write("\n");
    myfile.write("train: " + train_start + "-" + train_end + ", test: " + test_start + "-" + test_end);
    myfile.write("\n");
    myfile.write("sampling: " + str(sampling_rate) + ", period: " + str(feature_length));
    myfile.write("\n");
    myfile.write(result_r);
    myfile.write("\n");