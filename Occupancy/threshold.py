import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import os
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from datetime import datetime as dt
import sys
import argparse

# Constants
START_IDX = 21600; # 6 AM
END_IDX = 79200; # 10PM
DELTA = 10.0; # >10 watts indicates on off events, based on tablet charger power consumption (the smallest)

def perf_measure(y_actual, y_pred):
  TP, FP, TN, FN = 0,0,0,0;
  total_sample = 0;
  precision, recall, F = 0, 0, 0;
  for i in range(len(y_pred)):
    if y_actual[i]==y_pred[i]==1:
      TP += 1;
  for i in range(len(y_pred)): 
    if y_actual[i]==0 and y_actual[i]!=y_pred[i]:
      FP += 1;
  for i in range(len(y_pred)): 
    if y_actual[i]==y_pred[i]==0:
      TN += 1;
  for i in range(len(y_pred)):
    if y_actual[i]==1 and y_actual[i]!=y_pred[i]:
      FN += 1;
  print("TP: %s" % TP);
  print("FP: %s" % FP);
  print("TN: %s" % TN);
  print("FN: %s" % FN);
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
  print("precision: %s" % precision);
  print("recall: %s" % recall);
  print("F: %s" % F);
  return (TP, FP, TN, FN, precision, recall, F);

  
def read_occupancy(occ_filename, dates):
  occ_raw = pd.read_csv(filepath_or_buffer=occ_path + occ_filename, skiprows=0, sep=',');
  occ_data = pd.DataFrame(data=None, columns=occ_raw.columns);
  for date in dates:
    idx = occ_raw['Unnamed: 0'].str.contains(date);
    occ_data = occ_data.append(occ_raw[idx]);
  occ_data = occ_data.drop(occ_data.columns[0:START_IDX+1], axis=1); # can take from 06:00:00
  occ_data = occ_data.drop(occ_data.columns[END_IDX-START_IDX:], axis=1); # can take until 21:59:59
  return occ_data;


# find average occupancy for every feature length
def label_occupancy(occ_data):
  occ_label = occ_data[occ_data.columns[0:feature_length]].mean(axis=1).to_frame();
  occ_label.columns = ['6'];
  for it in range(1, timeslot):
    idx = it * feature_length;
    occ_mean = occ_data[occ_data.columns[idx:idx+feature_length]].mean(axis=1).to_frame();
    occ_mean.columns = [str(it + 6)];
    occ_label = pd.concat([occ_label, occ_mean], axis=1);
  occ_label_round = occ_label.round();
  occ_label_stack = occ_label_round.stack();
  return occ_label_stack;

 
# load data from CSV files inside the directory
def load_data(dir_path):
  a_data = []; # alltime data; d_data is daily data
  dates = [];
  files = os.listdir(sm_path);
  for i in files:
    if i.endswith(".csv"):
      date = dt.strptime(i, '%Y-%m-%d.csv');
      dates.append(date.strftime('%d-%b-%Y'));
      # read the data from 6 AM to 10 PM (data 21600 to 79200)
      d_data = pd.read_csv(filepath_or_buffer = sm_path + i, header=None, sep=',', usecols=[0,1,2]);
      d_data = d_data[START_IDX:END_IDX:1];
      # d_data = d_data.rolling(sampling_rate).mean();
      d_data = d_data[0::sampling_rate];
      a_data.append(d_data);
  a_data = pd.concat(a_data);
  return dates, a_data;

  
# extract features from raw_data
def extract_features(raw_data, occ_data, dates):
  a_features = [];
  slot_depart = 3 * (3600 / feature_length); # 9AM (6 + 3)
  slot_arrive = 11 * (3600 / feature_length); # 5PM (6 + 11)
  dropped_cols = [0,1];
  timestamps = [];
  for day in dates:
    date = dt.strptime(day, '%d-%b-%Y');
    datestamp = date.strftime('%m/%d/%Y');
    timestamp = pd.date_range(datestamp, periods=timeslot, freq=feature_freq).values;
    timestamps.append(timestamp);			
  timestamps = np.asarray(timestamps).flatten().flatten();
  raw_data = raw_data[0::feature_length/sampling_rate];
  raw_data = raw_data.reset_index().drop('index', axis=1);
  start_time = time.time();	
  raw_data[3] = raw_data.sum(axis=1);
  raw_data = raw_data.set_index(timestamps).tshift(6, freq='H');
  timestamp = raw_data.index.values;
  return raw_data, timestamp;


## TRAINING PHASE
# default value
sampling_rate = 1; # in seconds
OCCUPANCY_THRESHOLD = 100.0;
feature_length = 900; # in seconds, defaults to 15 minutes
house = 'r2';
parser = argparse.ArgumentParser();
parser.add_argument("--house", help="House dataset to be used. Could be r1, r2, r3.");
parser.add_argument("--sr", help="Resampling rate, from 1 second to 15 minutes.");
parser.add_argument("--ot", help="Power threshold to determine occupancy");
parser.add_argument("--fl", help="Length of raw data required to compute a single feature point. ETH paper default is 15 minutes/900 raw data.");
args = parser.parse_args();

if args.sr:
  sampling_rate = int(args.sr); # in seconds

if args.house:
  house = args.house;

if args.ot:
  OCCUPANCY_THRESHOLD = int(args.ot);
  
if args.fl:
  feature_length = int(args.fl); # in seconds	
	
if house=='r1':
  sm_path = '../../dataset/01_sm_csv/01_cross/';
  occ_path = '../../dataset/01_occupancy_csv/';
  occ_file = '01_summer.csv';
elif house=='r2':
  sm_path = '../../dataset/02_sm_csv/02_cross_month/';
  occ_path = '../../dataset/02_occupancy_csv/';
  occ_file = '02_summer.csv';
elif house=='r3':
  sm_path = '../../dataset/03_sm_csv/03_cross/';
  occ_path = '../../dataset/03_occupancy_csv/';
  occ_file = '03_summer.csv';
else:
  print ("house is not recognized. should be r1, r2, or r3");
  sys.exit();

if (feature_length % sampling_rate) > 1:
  print ("feature length must be divisible by, minimum twice, sampling_rate. exiting program...");
  sys.exit();
	
sampleslot = (END_IDX-START_IDX)/sampling_rate;
timeslot = (END_IDX-START_IDX)/feature_length;
max_sampling_idx = feature_length/sampling_rate;
sample_freq = str(sampling_rate) + 's';
feature_freq = str(feature_length) + 's';

# load data
start_time = time.time();
dates, a_data = load_data(sm_path);
#print("--- load training data: %s seconds ---" % (time.time() - start_time));

# create ground truth data
start_time = time.time();
occ_data = read_occupancy(occ_file, dates);
total_occ_label = label_occupancy(occ_data);
#print("--- load occ_training_label: %s seconds ---" % (time.time() - start_time));

# extract features
start_time = time.time();
total_sm, total_timestamps = extract_features(a_data, occ_data, dates);
timestamps = np.array(total_timestamps);
occ_label = np.array(total_occ_label);
#print("--- extract all_features: %s seconds ---" % (time.time() - start_time));

test = ['2012-08-20','2012-08-21','2012-08-22','2012-08-23','2012-08-25','2012-08-26'];
#test = ['2012-08-20'];
train_jun = ['2012-06-02','2012-06-03','2012-06-04','2012-06-05','2012-06-06','2012-06-07','2012-06-10','2012-06-11','2012-06-13','2012-06-16','2012-06-17','2012-06-18','2012-06-20','2012-06-22','2012-06-24','2012-06-25','2012-06-26','2012-06-28','2012-06-30','2012-06-02','2012-06-03','2012-06-04','2012-06-05','2012-06-06','2012-06-07','2012-06-10','2012-06-11','2012-06-13','2012-06-16','2012-06-17','2012-06-18','2012-06-20','2012-06-22','2012-06-24','2012-06-25','2012-06-26','2012-06-28','2012-06-30'];
train_jul = ['2012-07-10','2012-07-11','2012-07-13','2012-07-14','2012-07-16','2012-07-17','2012-07-19','2012-07-20','2012-07-22','2012-07-23','2012-07-24','2012-07-25'];
train_aug = ['2012-08-02','2012-08-04','2012-08-06','2012-08-07','2012-08-09','2012-08-11','2012-08-12','2012-08-14','2012-08-15','2012-08-16','2012-08-18','2012-08-27','2012-08-29','2012-08-30'];
train_all = train_jun + train_jul + train_aug;

X_test = pd.DataFrame(columns=total_sm.columns);
X_train = pd.DataFrame(columns=total_sm.columns);
y_train, y_test, timestamps_train, timestamps_test = [], [], [], [];

for idx, timestamp in enumerate(timestamps):
  if str(timestamp)[0:10] in test:
    X_test = X_test.append(total_sm.iloc[idx]);
    y_test.append(occ_label[idx]);
    timestamps_test.append(timestamp);
  elif str(timestamp)[0:10] in train_all:
    X_train = X_train.append(total_sm.iloc[idx]);
    y_train.append(occ_label[idx]);
    timestamps_train.append(timestamp);

## TESTING PHASE
prediction = X_test[3]; # can be phase 1 (a[0]), 2, 3, or all
prediction[prediction<=OCCUPANCY_THRESHOLD] = 0;
prediction[prediction>OCCUPANCY_THRESHOLD] = 1;
prediction_df = pd.DataFrame(data=prediction, index=timestamps_test);
y_test_df = pd.DataFrame(data=y_test, index=timestamps_test);

# group prediction by date
mispred_df = abs(prediction_df - y_test_df);
mispred_grouped = mispred_df.groupby(mispred_df.index.map(lambda x:str(x)[0:10])).mean();
accuracy_grouped = mispred_grouped.apply(lambda x: 1-x);
# accuracy_grouped.to_csv("accuracy_grouped.csv")
accuracy = accuracy_grouped.mean()[0];
print ("accuracy avg doubled: %s" % accuracy);
TP, FP, TN, FN, precision, recall, F = perf_measure(y_test, prediction);

result = house + "," + str(sampling_rate) + "," + str(feature_length) + "," + str(accuracy) + "," + str(TP) + "," + str(FP) + "," + str(TN) + "," + str(FN) + "," + str(precision) + "," + str(recall) + "," + str(F);
with open("result_thres_phaseall_week_metrics.csv", "a") as myfile:
    myfile.write("\n");
    myfile.write(result);