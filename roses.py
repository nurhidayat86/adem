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
from Occupancy import pcasvmconf as occ
from NILMTK.angga import nilmtkECOappliance as nil
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
  train_features['Occ'] = occupancy_ground_truth;
  train_features['Tablet'] = appliance_power_ground_truth.ix[:,0];
  train_features['Dishwasher'] = appliance_power_ground_truth.ix[:,1];
  train_features['Air exhaust'] = appliance_power_ground_truth.ix[:,2];
  train_features['Fridge'] = appliance_power_ground_truth.ix[:,3];
  train_features['Entertainment'] = appliance_power_ground_truth.ix[:,4];
  train_features['Freezer'] = appliance_power_ground_truth.ix[:,5];
  train_features['Kettle'] = appliance_power_ground_truth.ix[:,6];
  train_features['Lamp'] = appliance_power_ground_truth.ix[:,7];
  train_features['Laptops'] = appliance_power_ground_truth.ix[:,8];
  train_features['Stove'] = appliance_power_ground_truth.ix[:,9];
  train_features['TV'] = appliance_power_ground_truth.ix[:,10];
  train_features['Stereo'] = appliance_power_ground_truth.ix[:,11];
  test_features = pd.concat([sm_test, group_mix_test], axis=1);
  test_features['Occ'] = occupancy_prediction;
  test_features['Tablet'] = appliance_power.ix[:,0];
  test_features['Dishwasher'] = appliance_power.ix[:,1];
  test_features['Air exhaust'] = appliance_power.ix[:,2];
  test_features['Fridge'] = appliance_power.ix[:,3];
  test_features['Entertainment'] = appliance_power.ix[:,4];
  test_features['Freezer'] = appliance_power.ix[:,5];
  test_features['Kettle'] = appliance_power.ix[:,6];
  test_features['Lamp'] = appliance_power.ix[:,7];
  test_features['Laptops'] = appliance_power.ix[:,8];
  test_features['Stove'] = appliance_power.ix[:,9];
  test_features['TV'] = appliance_power.ix[:,10];
  test_features['Stereo'] = appliance_power.ix[:,11];
  return train_features, test_features;

####################  
# START OF PROGRAM #
####################

sampling_rate = 1; # in seconds
feature_length = 60; # in seconds, defaults to 15 minutes
dataset_loc = '../dataset/eco.h5';
train_start = "2012-06-02";
train_end = "2012-06-09";
train_end_nil = "2012-06-10";
test_start = "2012-06-10";
test_end = "2012-06-11";
test_end_nil = "2012-06-12";

parser = argparse.ArgumentParser();
parser.add_argument("--sr", help="Sampling rate");
parser.add_argument("--fl", help="Feature length");
parser.add_argument("--str", help="Start of train, format is YYYY-MM-DD");
parser.add_argument("--etr", help="End of train, format is YYYY-MM-DD");
parser.add_argument("--ste", help="Start of test, format is YYYY-MM-DD");
parser.add_argument("--ete", help="End of test, format is YYYY-MM-DD");
args = parser.parse_args();

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
# compute house level occupancy and aggregated smart meter features
occupancy_ground_truth, occupancy_prediction, sm_train, sm_test = occ.occupancy_sync_predict(train_start, train_end, test_start, test_end, sampling_rate, feature_length);
# compute disaggregated power per appliances
appliance_power, appliance_power_test_gt, co_model, appliance_power_ground_truth = nil.nilmtkECOfunc(dataset_loc, train_start, train_end_nil, test_start, test_end_nil, feature_length);
appliance_power.index.tz = None;
appliance_power = appliance_power.shift(periods=2, freq='H');
appliance_power_ground_truth.index.tz = None;
appliance_power_ground_truth = appliance_power_ground_truth.shift(periods=2, freq='H'); 
# compute which appliances are on or off based on grouping rules. also computes room level ground truth and number of people
train_ground_truth, group_mix_train = rlo.groupmix_rlo_generator(dataset_loc, train_start, train_end_nil, feature_length, occupancy_ground_truth, co_model); 
test_ground_truth, group_mix_test = rlo.groupmix_rlo_generator(dataset_loc, test_start, test_end_nil, feature_length, occupancy_prediction, co_model);
train_features, test_features = merge_features();
train_gt_room = train_ground_truth.drop('people', 1);
test_gt_room = test_ground_truth.drop('people', 1);
train_gt_people = train_ground_truth['people'];
test_gt_people = test_ground_truth['people'];

# drop NAs
train_features = train_features.dropna();
test_features = test_features.dropna();
train_gt_room = train_gt_room.loc[train_features.dropna().index];
test_gt_room = test_gt_room.loc[test_features.dropna().index];
train_gt_people = train_gt_people.loc[train_features.dropna().index];
test_gt_people = test_gt_people.loc[test_features.dropna().index];

# train multilabel SVM classifier
# classif_room = OneVsRestClassifier(svm.SVC(kernel='rbf'));
# classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));
# classif_people = OneVsRestClassifier(svm.SVC(kernel='rbf'));
# classif_people.fit(train_features.values.astype(float), train_gt_people.values.astype(int));

# train multilabel KNN
# classif_room = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree'));
# classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));
# classif_people = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree'));
# classif_people.fit(train_features.values.astype(float), train_gt_people.values.astype(int));

# train adaboost
# classif_room = OneVsRestClassifier(ensemble.AdaBoostClassifier());
# classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));
# classif_people = OneVsRestClassifier(ensemble.AdaBoostClassifier());
# classif_people.fit(train_features.values.astype(float), train_gt_people.values.astype(int));

# train random forest
classif_room = OneVsRestClassifier(ensemble.RandomForestClassifier());
classif_room.fit(train_features.values.astype(float), train_gt_room.values.astype(int));
classif_people = OneVsRestClassifier(ensemble.RandomForestClassifier());
classif_people.fit(train_features.values.astype(float), train_gt_people.values.astype(int));

# predict and get accuracy metrics
test_prediction_room = classif_room.predict(test_features);
test_prediction_people = classif_people.predict(test_features);

TP_r, FP_r, TN_r, FN_r, precision_r, recall_r, F_r = perf_measure_room(test_gt_room.values.astype(int), test_prediction_room);
TP_p, FP_p, TN_p, FN_p, precision_p, recall_p, F_p = perf_measure_people(test_gt_people.values.astype(int), test_prediction_people);

result_r = "room: " + str(TP_r) + "," + str(FP_r) + "," + str(TN_r) + "," + str(FN_r) + "," + str(precision_r) + "," + str(recall_r) + "," + str(F_r);
result_p = "people: " + str(feature_length) + "," + str(TP_p) + "," + str(FP_p) + "," + str(TN_p) + "," + str(FN_p) + "," + str(precision_p) + "," + str(recall_p) + "," + str(F_p);

with open("result_roses.csv", "a") as myfile:
    myfile.write("\n");
    myfile.write("train: " + train_start + "-" + train_end + ", test: " + test_start + "-" + test_end);
    myfile.write("\n");
    myfile.write("sampling: " + str(sampling_rate) + ", period: " + str(feature_length));
    myfile.write("\n");
    myfile.write(result_r);
    myfile.write("\n");
    myfile.write(result_p);
    myfile.write("\n");