# possible multilabel classifier: SVM and its estimators, Nearest Neighbours
# Explore scikit-multilearn (includes Meka wrapper)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from nilmtk import Dataset
import os
import time
from datetime import datetime as dt
import sys
import argparse
import pcasvmconf as occ
import nilmtkECOappliance as nil
import RLO as rlo

def perf_measure(test_ground_truth, test_prediction):
  TP, FP, TN, FN, total_sample, precision, recall, F = 0, 0, 0, 0, 0, 0, 0, 0;
  num_predictions = test_prediction.shape[0];
  num_rooms = test_prediction.shape[1];
  
  for i in range(num_predictions):
    for j in range(num_rooms):
      if test_ground_truth[i][j]==test_prediction[i][j]==1:
        TP += 1;
      elif test_ground_truth[i][j]==0 and test_prediction[i][j]==1:
        FP += 1;
      elif test_ground_truth[i][j]==test_prediction[i][j]==0:
        TN += 1;
      elif test_ground_truth[i][j]==1 and test_prediction[i][j]==0:
        FN += 1;

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
  
def extract_ground_truth():
  ## RLO - NEED TO GET ROOM LEVEL GROUND TRUTH
  train_ground_truth = rlo.groundtruth_generator(, train_start, train_end, ); 
  test_ground_truth = rlo.groundtruth_generator(, predict_start, predict_end, );
  return train_ground_truth, test_ground_truth;
  
# extract smart meter max and avg, appliances powers, house level occupancy, and appliance group using predictive methods
def extract_features():
  # compute house level occupancy feature for training and testing. returns dataframe
  occupancy_ground_truth, occupancy_prediction, sm_train, sm_test = occ.occupancy_sync_predict(train_start, train_end, predict_start, predict_end, sampling_rate, feature_length);
  appliance_power_ground_truth, appliance_power = nil.nilmtkECO(train_start, train_end, predict_start, predict_end, feature_length);

  ## RLO - NEED TO GET GROUPING FEATURES
  group_ground_truth = ;
  group = ;
  
  train_features = sm_train;
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
  train_features['Groups'] = group_ground_truth;

  test_features = sm_test;
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
  test_features['Groups'] = group;

  return train_features, test_features;
	
parser = argparse.ArgumentParser();
parser.add_argument("--sr", help="Sampling rate");
parser.add_argument("--fl", help="Feature length");
parser.add_argument("--str", help="Start of train, format is YYYY-MM-DD");
parser.add_argument("--etr", help="End of train, format is YYYY-MM-DD");
parser.add_argument("--ste", help="Start of test, format is YYYY-MM-DD");
parser.add_argument("--ete", help="End of test, format is YYYY-MM-DD");
args = parser.parse_args();

sampling_rate = args.sr;
feature_length = args.fl;
train_start = args.str;
train_end = args.etr;
test_start = args.ste;
test_end = args.ete;

# extract features and ground truth for both training and testing
train_ground_truth, test_ground_truth = extract_ground_truth();
train_features, test_features = extract_features();
train_mlb = MultiLabelBinarizer().fit(train_ground_truth);
test_mlb = MultiLabelBinarizer().fit(test_ground_truth);

# train multilabel SVM classifier
classif = OneVsRestClassifier(SVC(kernel='rbf'));
classif.fit(train_features, train_mlb.transform(train_ground_truth));

# predict and get accuracy metrics
test_prediction = classif.predict(test_features);
TP, FP, TN, FN, precision, recall, F = perf_measure(test_ground_truth, test_prediction);

result = str(sampling_rate) + "," + str(feature_length) + "," + str(TP) + "," + str(FP) + "," + str(TN) + "," + str(FN) + "," + str(precision) + "," + str(recall) + "," + str(F);
print result;