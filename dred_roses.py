# possible multilabel classifier: SVM, Nearest Neighbours, Decision Trees, Random Forest
# Explore scikit-multilearn (includes Meka wrapper)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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
from Occupancy import dred_occupancy as da
from NILMTK.angga import nilmtkDREDappliance as nil
from RLO import RLO as rlo

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
  train_features = pd.concat([sm_train, group_mix_train], axis=1);
  train_features[0] = appliance_power_ground_truth.ix[:,0];
  train_features[1] = appliance_power_ground_truth.ix[:,1];
  train_features[2] = appliance_power_ground_truth.ix[:,2];
  train_features[3] = appliance_power_ground_truth.ix[:,3];
  train_features[4] = appliance_power_ground_truth.ix[:,4];
  train_features[5] = appliance_power_ground_truth.ix[:,5];
  train_features[6] = appliance_power_ground_truth.ix[:,6];
  train_features[7] = appliance_power_ground_truth.ix[:,7];
  train_features[8] = appliance_power_ground_truth.ix[:,8];
  train_features[9] = appliance_power_ground_truth.ix[:,9];
  train_features[10] = appliance_power_ground_truth.ix[:,10];
  train_features[11] = appliance_power_ground_truth.ix[:,11];
  test_features = pd.concat([sm_test, group_mix_test], axis=1);
  test_features[0] = appliance_power.ix[:,0];
  test_features[1] = appliance_power.ix[:,1];
  test_features[2] = appliance_power.ix[:,2];
  test_features[3] = appliance_power.ix[:,3];
  test_features[4] = appliance_power.ix[:,4];
  test_features[5] = appliance_power.ix[:,5];
  test_features[6] = appliance_power.ix[:,6];
  test_features[7] = appliance_power.ix[:,7];
  test_features[8] = appliance_power.ix[:,8];
  test_features[9] = appliance_power.ix[:,9];
  test_features[10] = appliance_power.ix[:,10];
  test_features[11] = appliance_power.ix[:,11];
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
train_end = "2015-07-12";
train_end_nil = "2015-07-13";

test_start = "2015-07-13";
test_end = "2015-07-14";
test_end_nil = "2015-07-15";

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
# compute room level occupancy ground truth and aggregated smart meter features
train_gt_room = da.ro_gt(train_start, train_end, feature_length);
test_gt_room = da.ro_gt(test_start, test_end, feature_length);
sm_train = da.get_smf(train_start, train_end, feature_length);
sm_test = da.get_smf(test_start, test_end, feature_length);
# compute disaggregated power per appliances
appliance_power, appliance_power_test_gt, co_model, appliance_power_ground_truth = nil.nilmtkDREDfunc(dataset_loc, train_start, train_end_nil, test_start, test_end_nil, feature_length);
# compute which appliances are on or off based on grouping rules. also computes room level ground truth and number of people
group_mix_train = rlo.groupmix_generator(dataset_loc, train_start, train_end_nil, feature_length, co_model);
group_mix_test = rlo.groupmix_generator(dataset_loc, test_start, test_end_nil, feature_length, co_model);

# only pick index which has room level occupancy ground truth
sm_train = sm_train.loc[train_gt_room.index];
sm_test = sm_test.loc[test_gt_room.index];
appliance_power_ground_truth.index.tz = None;
appliance_power_ground_truth = appliance_power_ground_truth.shift(periods=2, freq='H'); 
appliance_power_ground_truth = appliance_power_ground_truth.loc[train_gt_room.index];
appliance_power.index.tz = None;
appliance_power = appliance_power.shift(periods=2, freq='H');
appliance_power = appliance_power.loc[test_gt_room.index];
group_mix_train.index.tz = None;
group_mix_train = group_mix_train.shift(periods=2, freq='H');
group_mix_train = group_mix_train.loc[train_gt_room.index];
group_mix_test.index.tz = None;
group_mix_test = group_mix_test.shift(periods=2, freq='H');
group_mix_test = group_mix_test.loc[test_gt_room.index];

train_features, test_features = merge_features();

# drop NAs
train_features = train_features.dropna();
test_features = test_features.dropna();
train_gt_room = train_gt_room.loc[train_features.dropna().index];
test_gt_room = test_gt_room.loc[test_features.dropna().index];

classif_room = OneVsRestClassifier(svm.SVC(kernel='rbf'));

# train multilabel SVM classifier
for i in range(0,7):
  if (i == 0):
    classif_room = OneVsRestClassifier(svm.SVC(kernel='rbf'));
  # train multilabel KNN
  elif (i == 1):
    classif_room = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree'));
  elif (i == 2):
    classif_room = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree'));
  elif (i == 3):
    classif_room = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'));
  elif (i == 4):
    classif_room = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree'));
  # train adaboost
  elif (i == 5):
    classif_room = OneVsRestClassifier(ensemble.AdaBoostClassifier());
  # train random forest
  elif (i == 6): 
    classif_room = OneVsRestClassifier(ensemble.RandomForestClassifier());
  classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));  
  # predict and get accuracy metrics
  test_prediction_room = classif_room.predict(test_features);
  TP_r, FP_r, TN_r, FN_r, precision_r, recall_r, F_r = perf_measure_room(test_gt_room.values.astype(int), test_prediction_room);
  result_r = "room: " + str(TP_r) + "," + str(FP_r) + "," + str(TN_r) + "," + str(FN_r) + "," + str(precision_r) + "," + str(recall_r) + "," + str(F_r);
  with open('Results' + os.path.sep + 'result_roses_dred.csv', "a") as myfile:
    myfile.write("\n");
    myfile.write("train: " + train_start + "-" + train_end + ", test: " + test_start + "-" + test_end);
    myfile.write("\n");
    myfile.write("sampling: " + str(sampling_rate) + ", period: " + str(feature_length));
    myfile.write("\n");
    myfile.write(result_r);
    myfile.write("\n");

