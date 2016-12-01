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

# merge smart meter max and avg, appliances powers, house level occupancy, and appliance group using predictive methods
def merge_features():
  train_features = pd.concat([sm_train, state_train], axis=1);
  train_features = sm_train;
  train_features = pd.DataFrame();
  train_features[0]  = appliance_power_train.ix[:,0];
  train_features[1]  = appliance_power_train.ix[:,1];
  train_features[2]  = appliance_power_train.ix[:,2];
  train_features[3]  = appliance_power_train.ix[:,3];
  train_features[4]  = appliance_power_train.ix[:,4];
  train_features[5]  = appliance_power_train.ix[:,5];
  train_features[6]  = appliance_power_train.ix[:,6];
  train_features[7]  = appliance_power_train.ix[:,7];
  train_features[8]  = appliance_power_train.ix[:,8];
  train_features[9]  = appliance_power_train.ix[:,9];
  train_features[10] = appliance_power_train.ix[:,10];
  train_features[11] = appliance_power_train.ix[:,11];
  test_features = pd.concat([sm_test, state_test], axis=1);
  test_features = sm_test;
  test_features[0]  = appliance_power_test.ix[:,0];
  test_features[1]  = appliance_power_test.ix[:,1];
  test_features[2]  = appliance_power_test.ix[:,2];
  test_features[3]  = appliance_power_test.ix[:,3];
  test_features[4]  = appliance_power_test.ix[:,4];
  test_features[5]  = appliance_power_test.ix[:,5];
  test_features[6]  = appliance_power_test.ix[:,6];
  test_features[7]  = appliance_power_test.ix[:,7];
  test_features[8]  = appliance_power_test.ix[:,8];
  test_features[9]  = appliance_power_test.ix[:,9];
  test_features[10] = appliance_power_test.ix[:,10];
  test_features[11] = appliance_power_test.ix[:,11];
  # normalize
  norm_f_train = train_features.dropna();
  min_max_scaler = preprocessing.MinMaxScaler();
  norm_ap_train = min_max_scaler.fit_transform(norm_f_train.values); # sometimes contains NaN
  norm_ap_train = pd.DataFrame(data=norm_ap_train, index=norm_f_train.index);
  norm_f_test = test_features.dropna();
  min_max_scaler = preprocessing.MinMaxScaler();
  norm_ap_test = min_max_scaler.fit_transform(norm_f_test.values); # sometimes contains NaN
  norm_ap_test = pd.DataFrame(data=norm_ap_test, index=norm_f_test.index);
  ## combine with state information
  #train_ap_state = pd.concat([state_train, train_features]);
  #test_ap_state = pd.concat([state_test, test_features]);
  return train_features, test_features;

####################  
# START OF PROGRAM #
####################

sampling_rate = 1; # in seconds
feature_length = 60; # in seconds, defaults to 15 minutes
classifier = 0;
dataset_loc = '../dataset/DRED.h5';

# training set is fairly distributed
train_start = "2015-07-06";
train_end = "2015-07-13";
train_end_nil = "2015-07-14";

test_start = "2015-08-01";
test_end = "2015-08-31";
test_end_nil = "2015-09-01";

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
# train_gt_room_ori = do.ro_gt(train_start, train_end, feature_length);
# test_gt_room_ori = do.ro_gt(test_start, test_end, feature_length);

# compute disaggregated power per appliances
appliance_power_test, appliance_power_test_gt, co_model, appliance_power_train = nil.nilmtkDREDfunc(dataset_loc, train_start, train_end_nil, test_start, test_end_nil, feature_length);
# remove timezone info from aggregated power dataframe
appliance_power_train.index.tz = None;
appliance_power_train = appliance_power_train.shift(periods=2, freq='H'); 
# appliance_power_test = pd.DataFrame.from_csv('fhmm_result.csv');
appliance_power_test.index.tz = None;
appliance_power_test = appliance_power_test.shift(periods=2, freq='H');

# compute room level occupancy ground truth and compute which appliances are on or off based on grouping rules
train_gt_room, state_train = rlo.occ_state_generator(dataset_loc, train_start, train_end_nil, feature_length, co_model);
test_gt_room, state_test = rlo.occ_state_generator(dataset_loc, test_start, test_end_nil, feature_length, co_model);
# remove timezone info from room level occupancy dataframe
train_gt_room.index.tz = None;
train_gt_room = train_gt_room.shift(periods=2, freq='H');
test_gt_room.index.tz = None;
test_gt_room = test_gt_room.shift(periods=2, freq='H');

# get binary state based on disaggregation model's states
appliance_power_test_state = appliance_power_test;
appliance_power_test_state.columns=['television','fan','fridge','laptop computer','electric heating element','oven','unknown','washing machine','microwave','toaster','sockets','cooker'];
state_test = rlo.state_generator(dataset_loc, test_start, test_end_nil, feature_length, co_model, appliance_power_test_state);
# remove timezone info from appliances state dataframe
state_train.index.tz = None;
state_train = state_train.shift(periods=2, freq='H');
state_test.index.tz = None;
state_test = state_test.shift(periods=2, freq='H');

# compute aggregated smart meter features
sm_train, sm_test = do.get_smf(train_start, train_end, test_start, test_end, feature_length);

## Only pick index which has room level occupancy ground truth
appliance_power_train = appliance_power_train.loc[train_gt_room.index];
appliance_power_test = appliance_power_test.loc[test_gt_room.index];

state_train = state_train.loc[train_gt_room.index];
state_test = state_test.loc[test_gt_room.index];

sm_train = sm_train.loc[train_gt_room.index];
sm_test = sm_test.loc[test_gt_room.index];

train_features, test_features = merge_features();

# drop NAs
train_features = train_features.dropna();
test_features = test_features.dropna();
train_gt_room = train_gt_room.loc[train_features.index];
test_gt_room = test_gt_room.loc[test_features.index];

# features_resampled, gt_room_resampled = sm.fit_sample(train_features, train_gt_room)
classif_room = OneVsRestClassifier(svm.SVC(kernel='rbf'));

#classif_room = OneVsRestClassifier(ensemble.AdaBoostClassifier());    
classif_room = OneVsRestClassifier(ensemble.RandomForestClassifier());
classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));  
# predict and get accuracy metrics
test_prediction_room = classif_room.predict(test_features);
precision_r, recall_r, F_r, support_r = precision_recall_fscore_support(y_true=test_gt_room.values.astype(int), y_pred=test_prediction_room, average='micro');
result_r = "classifier RF F score: " + str(F_r);
print result_r

#TP_r, FP_r, TN_r, FN_r, precision_r, recall_r, F_r = perf_measure_room(test_gt_room.values.astype(int), test_prediction_room);
#result_r = "room: " + str(TP_r) + "," + str(FP_r) + "," + str(TN_r) + "," + str(FN_r) + "," + str(precision_r) + "," + str(recall_r) + "," + str(F_r);
with open('Results' + os.path.sep + 'result_roses_dred.csv', "a") as myfile:
  myfile.write("\n");
  myfile.write("train: " + train_start + "-" + train_end + ", test: " + test_start + "-" + test_end);
  myfile.write("\n");
  myfile.write("sampling: " + str(sampling_rate) + ", period: " + str(feature_length));
  myfile.write("\n");
  myfile.write(result_r);
  myfile.write("\n");
  