import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
import os
import time
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit

from datetime import datetime as dt
import sys
import argparse
import pcasvmconf as occ

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
  return train_ground_truth, test_ground_truth;
  
# extract smart meter max and avg, appliances powers, house level occupancy, and appliance group using predictive methods
def extract_features():
  # compute house level occupancy feature for training and testing. returns dataframe
  occupancy_ground_truth, occupancy_prediction = occ.occupancy_sync_predict(train_start, train_end, predict_start, predict_end, sampling_rate, feature_length);
  
  train_features = pd.DataFrame();
  train_features['SM_max'] = ;
  train_features['SM_avg'] = ;
  train_features['Occ'] = 
  train_features['Tablet'] = 
  train_features['Dishwasher'] = 
  train_features['Air exhaust'] = 
  train_features['Fridge'] = 
  train_features['Entertainment'] = 
  train_features['Freezer'] = 
  train_features['Kettle'] = 
  train_features['Lamp'] = 
  train_features['Laptops'] =
  train_features['Stove'] = 
  train_features['TV'] =
  train_features['Stereo'] =
  train_features['Groups'] = ;

  test_features = pd.DataFrame();
  test_features['SM_max'] = ;
  test_features['SM_avg'] = ;
  test_features['Occ'] = 
  test_features['Tablet'] = 
  test_features['Dishwasher'] = 
  test_features['Air exhaust'] = 
  test_features['Fridge'] = 
  test_features['Entertainment'] = 
  test_features['Freezer'] = 
  test_features['Kettle'] = 
  test_features['Lamp'] = 
  test_features['Laptops'] =
  test_features['Stove'] = 
  test_features['TV'] =
  test_features['Stereo'] =
  test_features['Groups'] = ;

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