# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:09:04 2016

@author: neo
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
#from sklearn.grid_search import GridSearchCV
import os
import time
#import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn import cross_validation
#from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from datetime import datetime as dt
#from hmmlearn import hmm
from sknn.mlp import Classifier, Layer
import sys
import argparse
import pcasvmconf as psc
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact
from nilmtk import utils
import re;

# Constants
START_IDX = 21600; # 6 AM
END_IDX = 79200; # 10PM
DELTA = 20.0; # >30 watts indicates on off events

def get_states(CO):
    appliances_name = [];
    state0 = [];
    state1 = [];
    state2 = [];
    for i in range(0,len(CO.model)):
        appliances_name.append(str(re.search("type='(.*?)'",str(CO.model[i]['training_metadata'].appliances)).group(1)).lower());
        state0.append(CO.model[i]['states'][0]);
        if(len(CO.model[i]['states'])==3):
            state1.append(CO.model[i]['states'][1]);
            state2.append(CO.model[i]['states'][2]);
        elif(len(CO.model[i]['states'])==2):
            state1.append(CO.model[i]['states'][1]);
            state2.append(CO.model[i]['states'][1]);
        else:
            state1.append(CO.model[i]['states'][0]);
            state2.append(CO.model[i]['states'][0]);
    state = pd.DataFrame(index=appliances_name, columns=['state0','state1','state2']);
    state.ix[:,'state0'] = state0;
    state.ix[:,'state1'] = state1;
    state.ix[:,'state2'] = state2;
    state.to_csv('states.csv');
    return state;
        

def enhance_rule(housing, room_path, prediction_df, state):
    print("total data: " + str(len(prediction)));
    y_nilmtk = pd.DataFrame.from_csv(room_path);
    yout = y_nilmtk.join(prediction_df, how='inner');
    yout.to_csv('join_'+room_path);
#    y_on_off = yout;
    y_on_off = pd.DataFrame(columns=yout.columns,index=yout.index);
    y_on_off = y_on_off.drop('prediction', axis=1)
    group = [];
    single = [];
    for i in y_on_off.index:
        for j in y_on_off.columns:
            if(yout.ix[i,j] > 0):
                if (yout.ix[i,j]>=state.ix[j,'state1']):
                    y_on_off.ix[i,j] = 1;
                else:
                    y_on_off.ix[i,j] = 0;
            else:
                y_on_off.ix[i,j] = 0;
    y_on_off.to_csv('y_on_off'+room_path);
    if housing == 'r2':
        for i in y_on_off.index:
            #Looking for single / mix:
            mix = 0;
            if (y_on_off.ix[i,'kettle'] > 0) or (y_on_off.ix[i,'stove'] > 0) or (y_on_off.ix[i,'freezer'] > 0) or (y_on_off.ix[i,'fridge'] > 0) or (y_on_off.ix[i,'dish washer'] > 0):
                mix += 1;
            if (y_on_off.ix[i,'television'] > 0) or (y_on_off.ix[i,'audio system'] > 0) or (y_on_off.ix[i,'htpc'] > 0) or (y_on_off.ix[i,'lamp'] > 0):
                mix += 1;
            if (y_on_off.ix[i,'laptop computer'] > 0) or (y_on_off.ix[i,'air handling unit'] > 0) or (y_on_off.ix[i,'tablet computer charger'] > 0):
                mix += 1;
            if mix >= 2:
                single.append(True);
            else:
                single.append(False);
            
            #Looking for appliances group;
            sub_group = [];
            if (y_on_off.ix[i,'air handling unit'] > 0):
                sub_group.append('fridge');
                #sub_group.append('htpc');
                sub_group.append('freezer');
                #sub_group.append('audio system');
            if (y_on_off.ix[i,'audio system'] > 0):
                #sub_group.append('fridge');
                sub_group.append('htpc');
                sub_group.append('freezer');
                #sub_group.append('television');
            if (y_on_off.ix[i,'kettle'] > 0):
                #sub_group.append('fridge');
                sub_group.append('freezer');
            if (y_on_off.ix[i,'television'] > 0):
                sub_group.append('htpc');
                sub_group.append('freezer');
                sub_group.append('audio system');
            if (y_on_off.ix[i,'dish washer'] > 0):
                #sub_group.append('fridge');
                sub_group.append('freezer');
            #if (y_on_off.ix[i,'freezer'] > 0):
                #sub_group.append('fridge');
            if (y_on_off.ix[i,'fridge'] > 0):
                sub_group.append('freezer');
            if (y_on_off.ix[i,'htpc'] > 0):
                #sub_group.append('fridge');
                sub_group.append('freezer');
                sub_group.append('audio system');
            if (y_on_off.ix[i,'lamp'] > 0):
                #sub_group.append('fridge');
                sub_group.append('freezer');
                sub_group.append('htpc');
                sub_group.append('audio system');
                sub_group.append('television');
            if (y_on_off.ix[i,'laptop computer'] > 0):
                #sub_group.append('fridge');
                sub_group.append('freezer');
                #sub_group.append('htpc');
                #sub_group.append('fridge');
                #sub_group.append('audio system');
            if (y_on_off.ix[i,'stove'] > 0):
                sub_group.append('freezer');
            if (y_on_off.ix[i,'tablet computer charger'] > 0):
                #sub_group.append(fridge');
                sub_group.append('freezer');
            sub_group = list(set(sub_group));
            group.append(sub_group);
    yout.to_csv('yout_'+room_path);
    return single, group, yout, y_on_off;

def room_groundtruth(state, yout, housing, room_path):
    result = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'],index=yout.index);   
    if housing == 'r2':
        for i in yout.index:
            result.ix[i,:] = 0;
            if (yout.ix[i,'prediction'] < 1):
                print("Not at home");
            else:
                if (yout.ix[i,'kettle'] > 0) or (yout.ix[i,'stove'] > 0) or (yout.ix[i,'freezer'] >= int(state.ix['freezer','state2'])) or (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])) or (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                    result.ix[i,'kitchen'] = 1;
                if (yout.ix[i,'television'] >= int(state.ix['television','state2'])) or (yout.ix[i,'audio system'] >= int(state.ix['audio system','state2'])) or (yout.ix[i,'htpc'] >= int(state.ix['htpc','state2'])) or (yout.ix[i,'lamp'] > int(state.ix['lamp','state2'])):
                    result.ix[i,'livingroom'] = 1;
                if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])) or (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])) or (yout.ix[i,'tablet computer charger'] >= int(state.ix['tablet computer charger','state2'])):
                    result.ix[i,'bedroom'] = 1;           
                
                result.ix[i,'people'] = result.ix[i,'kitchen'] + result.ix[i,'livingroom'] + result.ix[i,'bedroom'] + result.ix[i,'bathroom'];
                
                if (result.ix[i,'people'] == 0):
#                    print('Somewhere');
                    result.ix[i,'people'] = 1;
    result.to_csv('groundtruth_room_'+room_path);
    return result;
                    
#    elif housing == 'r1':
#        for i in yout.index.values:
#            if (yout.ix[i,'prediction'] < 1):
#                print('Not at home');
#                result.ix[i,'people'] = 0;
#            else:
#                if (yout[i,'Hair dryer'] > 0) or (yout[i,'Washing machine'] > 1000):
#                    result.ix[i,'bathroom'] = 1;
#                if (yout[i,'Fridge'] > 30) or (yout[i,'Coffee maker'] > 100) or (yout[i,'Kettle'] > 250) or (yout[i,'Freezer'] > 25):
#                    result.ix[i,'kitchen'] = 1;
#                if (yout[i,'Computer'] > 100):
#                    result.ix[i,'bedroom'] = 1;
#                result.ix[i,'people'] = result.ix[i,'kitchen'] + result.ix[i,'livingroom'] + result.ix[i,'bedroom'] + result.ix[i,'bathroom'];
#                if (result.ix[i,'people'] == 0):
#                    print('Somewhere');
#                    result.ix[i,'people'] = 1;
#    elif housing == 'r3':
#        for i in yout.index.values:
#            if (yout.ix[i,'prediction'] < 1):
#                print('Not at home');
#                result.ix[i,'people'] = 0;
#            else:
#                if (yout[i,'Fridge'] > 40) or (yout[i,'Coffee maker'] > 100) or (yout[i,'Kettle'] > 250) or (yout[i,'Freezer'] > 18):
#                    result.ix[i,'kitchen'] = 1;
#                if (yout[i,'Computer'] > 100) or (yout[i,'Laptop computer'] > 50):
#                    result.ix[i,'bedroom'] = 1;
#                if (yout[i,'HTPC'] > 30):
#                    result.ix[i,'livingroom'] = 1;
#                result.ix[i,'people'] = result.ix[i,'kitchen'] + result.ix[i,'livingroom'] + result.ix[i,'bedroom'] + result.ix[i,'bathroom'];
#                if (result.ix[i,'people'] == 0):
#                    print('Somewhere');
#                    result.ix[i,'people'] = 1;
# load data from CSV files inside the directory

def load_data(dir_path, sampling_rate):
    a_data = []; # alltime data; d_data is daily data
    dates = [];
    dates2 = [];
    files = os.listdir(sm_path);
    files.sort();
    for i in files:
        if i.endswith(".csv"):
            date = dt.strptime(i, '%Y-%m-%d.csv');
            dates.append(date.strftime('%d-%b-%Y'));
            dates2.append(date.strftime('%m-%d-%Y'));
            # read the data from 6 AM to 10 PM (data 21600 to 79200)
            d_data = pd.read_csv(filepath_or_buffer = sm_path + i, header=None, sep=',', usecols=[0,1,2]);
            d_data = d_data[START_IDX-sampling_rate:END_IDX:1]
            d_data = d_data.rolling(sampling_rate).sum();
            d_data = d_data[sampling_rate::sampling_rate];
            a_data.append(d_data);
    return dates, a_data, dates2;

def check_onff(x):
	if (abs(x) > DELTA):
		onoff = 1;
	else:
		onoff = 0;
	return onoff;

def calculate_sad(slot_data):
    abs_diff =  slot_data.diff().abs();
    return abs_diff.sum();

def calculate_onoff(slot_data):
	diff =  slot_data.diff();
	onoff = diff.applymap(check_onff);
	num_onoff = onoff.sum().values[0];
	return num_onoff;

def compute_feature(slot, slot_data):

    power123 = slot_data.ix[:,[0,1,2]].sum(axis=1);

    min1 = slot_data.ix[:,[0]].min().values[0];
    min2 = slot_data.ix[:,[1]].min().values[0];
    min3 = slot_data.ix[:,[2]].min().values[0];
    min123 = power123.min();
    max1 = slot_data.ix[:,[0]].max().values[0];
    max2 = slot_data.ix[:,[1]].max().values[0];
    max3 = slot_data.ix[:,[2]].max().values[0];
    max123 = power123.max();
    mean1 = slot_data.ix[:,[0]].mean().values[0];
    mean2 = slot_data.ix[:,[1]].mean().values[0];
    mean3 = slot_data.ix[:,[2]].mean().values[0];
    mean123 = power123.mean();
    std1 = slot_data.ix[:,[0]].std().values[0];
    std2 = slot_data.ix[:,[1]].std().values[0];
    std3 = slot_data.ix[:,[2]].std().values[0];
    std123 = power123.std();
    sad1 = calculate_sad(slot_data.iloc[:,[0]]).values[0];
    sad2 = calculate_sad(slot_data.iloc[:,[1]]).values[0];
    sad3 = calculate_sad(slot_data.iloc[:,[2]]).values[0];
    sad123 = calculate_sad(slot_data.iloc[:,[0,1,2]].sum(axis=1).to_frame()).values[0];
    corl1 = slot_data.ix[:,0].autocorr();
    corl2 = slot_data.ix[:,1].autocorr();
    corl3 = slot_data.ix[:,2].autocorr();
    corl123 = power123.autocorr();
    onoff1 = calculate_onoff(slot_data.iloc[:,[0]]);
    onoff2 = calculate_onoff(slot_data.iloc[:,[1]]);
    onoff3 = calculate_onoff(slot_data.iloc[:,[2]]);
    onoff123 = calculate_onoff(slot_data.iloc[:,[0,1,2]].sum(axis=1).to_frame());
    range1 = max1 - min1;
    range2 = max2 - min2;
    range3 = max3 - min3;
    range123 = max123 - min123;
    ptime = slot+1;
    # 9AM to 5PM, slot 12 to 47
    if slot >= 12 and slot <=47 :
        pfixed = 1;
    else:
        pfixed = 0;
    feature = [min1, min2, min3, min123, max1, max2, max3, max123, mean1, mean2, mean3, mean123, std1, std2, std3, std123, sad1, sad2, sad3, sad123, corl1, corl2, corl3, corl123, onoff1, onoff2, onoff3, onoff123, range1, range2, range3, range123, pfixed, ptime];
    return feature;

def compute_pprob(occ_data, sampling_rate, feature_length):
	max_sampling_idx = feature_length/sampling_rate;
	occ_prob = occ_data[occ_data.columns[0:max_sampling_idx]].sum(axis=1).to_frame();
	timeslot = (END_IDX-START_IDX)/feature_length;
	for it in range(1, timeslot):
		idx = it * max_sampling_idx;
		occ_sum = occ_data[occ_data.columns[idx:idx+max_sampling_idx]].sum(axis=1).to_frame();
		occ_prob = pd.concat([occ_prob, occ_sum], axis=1);
	occ_prob = occ_prob.div(max_sampling_idx);
	occ_prob = occ_prob.stack();
	return occ_prob;

# extract features from raw_data
def extract_features(raw_data, occ_data, occ_label, dates, sampling_rate, feature_length):
	a_features = [];
	max_sampling_idx = feature_length/sampling_rate;
	timeslot = (END_IDX-START_IDX)/feature_length;
	curr_freq = str(feature_length) + 's';

	for idx, day in enumerate(raw_data):
		date = dt.strptime(dates[idx], '%d-%b-%Y');
		datestamp = date.strftime('%Y/%m/%d');
		timestamp = pd.date_range(datestamp, periods=timeslot, freq=curr_freq);
		day = day[day >= 0]; # remove data with negative power value
		d_features = pd.DataFrame(columns=('min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'pfixed', 'ptime'));
		# e.g. feature length 15 minutes, then iterate over 16 * 4 slots = 64 slots
		for slot in range(0, timeslot):
			idx = slot * max_sampling_idx;
			d_features.loc[slot] = compute_feature(slot, day[idx:idx+max_sampling_idx:1]);
		d_features  = d_features.set_index(timestamp).tshift(6, freq='H');
		a_features.append(d_features);

	total_features = pd.concat(a_features);
	#total_features = total_features.reset_index();
	#total_features = total_features.drop('index', 1);
	pprob = compute_pprob(occ_data, sampling_rate, feature_length).to_frame();
	pprob = pprob.reset_index();
	dropped_cols = [0,1];
	pprob = pprob.drop(pprob.columns[dropped_cols], 1);
	total_features['pprob'] = pd.Series(pprob.ix[:,0].tolist(), index=total_features.index);
	timestamp = total_features.index.values;
	
	# normalize data. should be on the features, not on the data!
	x = total_features.values; # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler();
	isnan_index = np.where(np.isnan(x));
	x = np.delete(x, isnan_index[0], axis=0);
	occ_label = occ_label.drop(occ_label.index[isnan_index[0]]);
	x_scaled = min_max_scaler.fit_transform(x); # sometimes contains NaN
	total_features = pd.DataFrame(x_scaled);
	total_features.columns = ['min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'pfixed', 'ptime', 'pprob'];
	return total_features, occ_label, timestamp;

def read_occupancy(occ_filename, occ_filename2, dates):
    occ_raw = pd.read_csv(filepath_or_buffer=occ_path + occ_filename, skiprows=0, sep=',');
    occ_raw2 = pd.read_csv(filepath_or_buffer=occ_path + occ_filename2, skiprows=0, sep=',');
    occ_data = pd.DataFrame(data=None, columns=occ_raw.columns);
    for date in dates:
        idx = occ_raw['Unnamed: 0'].str.contains(date);
        occ_data = occ_data.append(occ_raw[idx]);
    occ_data2 = occ_raw;
    occ_data2 = occ_data2.append(occ_raw2)
    occ_data = occ_data.drop(occ_data.columns[0:START_IDX+1], axis=1);
    occ_data = occ_data.drop(occ_data.columns[END_IDX-START_IDX:], axis=1);
    del occ_raw2;
    return occ_data, occ_data2;
	
# find average occupancy for every 15 minutes 
def label_occupancy(occ_data, feature_length):	
	occ_label = occ_data[occ_data.columns[0:feature_length]].mean(axis=1).to_frame();
	occ_label.columns = ['6'];
	timeslot = (END_IDX-START_IDX)/feature_length;
	for it in range(1, timeslot):
		idx = it * feature_length;
		occ_mean = occ_data[occ_data.columns[idx:idx+feature_length]].mean(axis=1).to_frame();
		occ_mean.columns = [str(it + 6)];
		occ_label = pd.concat([occ_label, occ_mean], axis=1);	
	occ_label = occ_label.round();
	occ_label = occ_label.stack();
	return occ_label;
 
def produce_occupancy(occ_data2, feature_length):
    indexed_occupancy = occ_data2.set_index(['Unnamed: 0']);
    occupancy_col = [];
    occupancy_index = [];
    for i in indexed_occupancy.index.values:
        for j in indexed_occupancy.columns.values:
            occupancy_index.append(pd.to_datetime(str(i)+ " " + str(j), format="%d-%b-%Y '%H:%M:%S'"));
            occupancy_col.append(indexed_occupancy.ix[i,j]);
    result = pd.DataFrame(index=occupancy_index,data=occupancy_col, columns=['prediction']);
    result = result.resample(str(feature_length) + 'S').mean().round(0);
    result.to_csv('resample_occupancy.csv');
    return result;
    
## TRAINING PHASE
# default value
sampling_rate = 1; # in seconds
test_ratio = 0.6;
feature_length = 900; # in seconds, defaults to 15 minutes
house = 'r2';

parser = argparse.ArgumentParser();
parser.add_argument("--house", help="House dataset to be used. Could be r1, r2, r3.");
parser.add_argument("--sr", help="Resampling rate, from 1 second to 15 minutes.");
parser.add_argument("--tr", help="Testing data ratio. Training data ratio is therefore 1 - tr.");
parser.add_argument("--fl", help="Length of raw data required to compute a single feature point. ETH paper default is 15 minutes/900 raw data.");
args = parser.parse_args();

if args.sr:
	sampling_rate = int(args.sr); # in seconds

if args.tr:
	test_ratio = float(args.tr);

if args.house:
	house = args.house;

if args.fl:
	feature_length = int(args.fl); # in seconds	
	
if house=='r1':
    sm_path = '/home/neo/ECO/01_sm_csv/01_cross/';
    occ_path = '/home/neo/ECO/01_occupancy_csv/';
    occ_file = '01_summer.csv';
    occ_file2 = '01_winter.csv';
    room_path = '/home/neo/data1.csv';
elif house=='r2':
    sm_path = '/home/neo/ECO/02_sm_csv/02_cross/';
    occ_path = '/home/neo/ECO/02_occupancy_csv/';
    occ_file = '02_summer.csv';
    occ_file2 = '02_winter.csv';
    room_path = '/home/neo/data2.csv';
elif house=='r3':
    sm_path = '/home/neo/ECO/03_sm_csv/03_cross/';
    occ_path = '/home/neo/ECO/03_occupancy_csv/';
    occ_file = '03_summer.csv';
    occ_file2 = '03_winter.csv';
    room_path = '/home/neo/data3.csv';
else:
	print ("house is not recognized. should be r1, r2, or r3");
	sys.exit();

if (feature_length % sampling_rate) > 1:
	print ("feature length must be divisible by, minimum twice, sampling_rate. exiting program...");
	sys.exit();
	
# load data
start_time = time.time();
dates, a_data, dates2 = load_data(sm_path, sampling_rate);
#print("--- load training data: %s seconds ---" % (time.time() - start_time));

# create ground truth data
start_time = time.time();
occ_data, occ_data2 = read_occupancy(occ_file, occ_file2, dates);

#Use this one to construct csv
#occ_truth = produce_occupancy(occ_data2, feature_length);

#Otherwise use this if you have one (for much faster process)
occ_truth = pd.DataFrame.from_csv('resample_occupancy.csv');
occ_label = label_occupancy(occ_data, feature_length);
#print("--- load occ_training_label: %s seconds ---" % (time.time() - start_time));

# extract features
start_time = time.time();
all_features, occ_label, timestamps = extract_features(a_data, occ_data, occ_label, dates, sampling_rate, feature_length);
#print("--- extract all_features: %s seconds ---" % (time.time() - start_time));

# cross validation
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_features, occ_label, test_size=0.4, random_state=0);

timestamps = np.array(timestamps);
sss = StratifiedShuffleSplit(occ_label, 3, test_size=test_ratio, random_state=0);
# kf = KFold(all_features.shape[0], shuffle=True, n_folds=2);
for train_index, test_index in sss:
    timestamps_train, timestamps_test = timestamps[train_index], timestamps[test_index];
    X_train, X_test = all_features.as_matrix()[train_index], all_features.as_matrix()[test_index];
    y_train, y_test = occ_label.as_matrix()[train_index], occ_label.as_matrix()[test_index];

# load the features into pca	
start_time = time.time();
pca = PCA();
pca.fit(X_train);
#print("--- load training data to PCA: %s seconds ---" % (time.time() - start_time));

# take only L components that make up the 95% variance
start_time = time.time();
num_comp = 0;
comp_sum = 0.0;
ev_sum = np.sum(pca.explained_variance_ratio_);
for ev in pca.explained_variance_ratio_:
	num_comp = num_comp+1;
	comp_sum = comp_sum+ev;
	if ((comp_sum / ev_sum) > 0.95):
		break;
#print("--- pca.n_components: %s ---" % num_comp);
#print("--- find 0.95 variance: %s seconds ---" % (time.time() - start_time));

# run PCA
start_time = time.time();
pca.n_components = num_comp;
all_features_reduced = pca.fit_transform(X_train);
#print("--- run PCA for training: %s seconds ---" % (time.time() - start_time));

# run SVM classifier
svc = svm.SVC(kernel='rbf');
start_time = time.time();
svc.fit(all_features_reduced, y_train);

## TESTING PHASE
start_time = time.time();
test_features_reduced = pca.fit_transform(X_test);
#print("--- run PCA for testing: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
prediction = svc.predict(test_features_reduced);
data_timestamp = [];
prediction_df = pd.DataFrame(index=timestamps_test,data=prediction, columns=['prediction']);
prediction_df.to_csv('prediction.csv');
print str(data_timestamp);

#print("--- prediction: %s seconds ---" % (time.time() - start_time));
acc = accuracy_score(y_test, prediction);
print str(acc);

result = house + "," + str(test_ratio) + "," + str(sampling_rate) + "," + str(feature_length) + "," + str(acc);
with open("result.csv", "a") as myfile:
    myfile.write("\n");
    myfile.write(result);
    myfile.close();

#run with HMM
nn = Classifier(
    layers=[
        Layer("Sigmoid", units=10),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=200)
nn.fit(all_features_reduced, y_train)

y_example = nn.predict(test_features_reduced)
acc2 = accuracy_score(y_test, y_example);
print str(acc2)

result2 = house + "," + str(test_ratio) + "," + str(sampling_rate) + "," + str(feature_length) + "," + str(acc2);
with open("result2.csv", "a") as myfile:
    myfile.write("\n");
    myfile.write(result2);
    myfile.close();

print(dates2);

#total = DataSet('/home/neo/NILMTK_experimental/eco1.h5')
train = DataSet('/home/neo/NILMTK_experimental/eco1.h5')
test = DataSet('/home/neo/NILMTK_experimental/eco1.h5')

if house == 'r1':
    train.set_window(start="01-08-2012", end="09-02-2012")
    test.set_window(start=str(min(dates2)) + " 00:00:00", end=str(max(dates2)) + " 23:45:00")
    building = 1;
elif house == 'r2':
    train.set_window(start="07-01-2012", end="09-30-2012")
    test.set_window(start=str(min(dates2)) + " 00:00:00", end=str(max(dates2)) + " 23:45:00")
    building = 2;
elif house == 'r3':
    train.set_window(start="11-01-2012", end="11-30-2012")
    test.set_window(start=str(min(dates2)) + " 00:00:00", end=str(max(dates2)) + " 23:45:00")
    building = 3;


train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec
tf_train = train.buildings[building].elec.mains().get_timeframe()
tf_test = test.buildings[building].elec.mains().get_timeframe()


#output with co training
co = CombinatorialOptimisation();
co.train(train_elec.submeters(), sample_period=feature_length);

#retrieving state database
state = get_states(co);

#build disaggregation
disag_filename_co = '/home/neo/NILMTK_experimental/disarg_folder/disarg_co.h5';
output = HDFDataStore(disag_filename_co, 'w');
co.disaggregate(test_elec.mains(), output, sample_period=feature_length);
output.close();

#capturing nilmtk result
disag_co = DataSet(disag_filename_co);
disag_elec = disag_co.buildings[2].elec;

#printing on CSV
nilmtk_csv = 'nilmtk_result.csv';
training_csv = 'nilmtk_train.csv';
disag_elec_df = disag_elec.dataframe_of_meters();
train_elec_df = train_elec.dataframe_of_meters().resample('900S').max().round(0);
disag_elec_df = disag_elec_df.drop(disag_elec_df.columns[0], axis=1);
train_elec_df = train_elec_df.drop(train_elec_df.columns[[0,1,2]], axis=1);
label = [];
for i in disag_elec.submeters().instance():
    label.append(str(disag_elec[i].label()).lower());
disag_elec_df.columns = label;
label = [];
for i in train_elec.submeters().instance():
    label.append(str(train_elec[i].label()).lower());
train_elec_df.columns = label;
disag_elec_df.to_csv(nilmtk_csv);
train_elec_df.to_csv(training_csv);

print("Done writing files")
print("start dates: " + str(min(dates2)));
print("max dates: " + str(max(dates2)));

#print("Constructing association rules -- do only if you need this data")
#ass_data = pd.DataFrame.from_csv('./train900.csv');
#ass_data = ass_data.join(occ_truth);
#ass_data.to_csv('ass_data.csv');

single_test, group_test, yout_test, y_on_off_test = enhance_rule(house, nilmtk_csv, prediction_df, state);
single_train, group_train, yout_train, y_on_off_train = enhance_rule(house, training_csv, occ_truth, state);

#building groundtruth room occupancy data
ground_truth_test = room_groundtruth(state, yout_test, house, nilmtk_csv);
ground_truth_train = room_groundtruth(state, yout_train, house, training_csv);

#Training with room level occupancy

#result.to_csv('result_close.csv');
#data = 'Rules, Support, Confidence \r\n';
#print('Preparing to write association rules');
#import Orange;
#assoc_data = Orange.data.Table('ass_data.csv');
#rules = Orange.associate.AssociationRulesInducer(assoc_data, support=0.2);
#for r in rules:
#    data += str("%s,  %5.3f,  %5.3f" % (r, r.support, r.confidence));
#    data += '\r\n'
#target = open('rules2.csv','w');
#target.write(data);
#target.close();



#printing on CSV
# write disaggregation output format 2 (colomn)
#nilmtk_csv = 'ground_truth.csv'
#target = open(nilmtk_csv, 'w');
#data = 'timestamp';
#data += '/t';    
#for instance in disag_elec.submeters().instance():
#    data += disag_elec[instance].label();            
#    data += '/t';
#    #print(data)
#data += '\r\n';
#target.write(data);
#
#size = disag_elec[1].load().next().axes[0].size;    
#for i in range(size):
#    data = '';
#    data += str(utils.convert_to_timestamp(disag_elec[instance].load().next().axes[0][i].value));
#    data += '/t';
#    for instance in disag_elec.submeters().instance():
#        data += str(disag_elec[instance].load().next().ix[i][0]);
#        data += '/t';        
#    data += '\r\n';
#    #print(data)
#    target.write(data);
#target.close();


#with open("result_close.csv", "a") as myfile:
#    data += "People, Kitchen, Bedroom, Bathroom, Livingroom\r\n"
#    for i in range(0, len(people)-1):    
#        data += str(people[i]) + "," + str(kitchen[i]) + "," + str(bedroom[i]) + "," + str(bathroom[i]) + "," + str(living_room[i]);
#        data += "\r\n";
#    myfile.write(data);
#myfile.close();