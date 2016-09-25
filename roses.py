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
from nilmtk import DataSet
import os
import time
from datetime import datetime as dt
import sys
import argparse
from Occupancy import pcasvmconf as occ
from NILMTK import nilmtkECOappliance as nil
from RLO import RLO as rlo

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
  ##NILMTK data
  disag_co_elec_submeter_df, gt_df_sub, co = nil.nilmtkECOfunc(train_start, train_end, test_start, test_end, feature_length);  
  occupancy_ground_truth, occupancy_prediction, sm_train, sm_test = occ.occupancy_sync_predict(train_start, train_end, test_start, test_end, sampling_rate, feature_length);
  
  ## RLO - NEED TO GET ROOM LEVEL GROUND TRUTH
  train_ground_truth, ml_input_groundtruth,test_ground_truth, ml_input_test = rlo.groundtruth_generator(gt_df_sub, disag_co_elec_submeter_df, 2, co, occupancy_ground_truth, occupancy_prediction); 
  return train_ground_truth, ml_input_groundtruth, test_ground_truth, ml_input_test;
  
	
parser = argparse.ArgumentParser();
parser.add_argument("--sr", help="Sampling rate");
parser.add_argument("--fl", help="Feature length");
parser.add_argument("--str", help="Start of train, format is YYYY-MM-DD");
parser.add_argument("--etr", help="End of train, format is YYYY-MM-DD");
parser.add_argument("--ste", help="Start of test, format is YYYY-MM-DD");
parser.add_argument("--ete", help="End of test, format is YYYY-MM-DD");
args = parser.parse_args();

sampling_rate = 1;#args.sr;
feature_length = 60;#args.fl;
train_start = '2012-06-02';#args.str;
train_end = '2012-06-09'#args.etr;
test_start = '2012-06-11'#args.ste;
test_end = '2012-06-12'#args.ete;

# extract features and ground truth for both training and testing
train_ground_truth, ml_input_groundtruth, test_ground_truth, ml_input_test = extract_ground_truth();
train_mlb = MultiLabelBinarizer().fit(train_ground_truth);
test_mlb = MultiLabelBinarizer().fit(test_ground_truth);

# train multilabel SVM classifier
classif = OneVsRestClassifier(SVC(kernel='rbf'));
classif.fit(ml_input_groundtruth, train_mlb.transform(train_ground_truth));

# predict and get accuracy metrics
test_prediction = classif.predict(ml_input_test);
TP, FP, TN, FN, precision, recall, F = perf_measure(test_ground_truth, test_prediction);

result = str(sampling_rate) + "," + str(feature_length) + "," + str(TP) + "," + str(FP) + "," + str(TN) + "," + str(FN) + "," + str(precision) + "," + str(recall) + "," + str(F);
print result;