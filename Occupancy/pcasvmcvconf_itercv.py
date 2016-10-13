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
	TP = 0;
	FP = 0;
	TN = 0;
	FN = 0;
	total_sample = 0;
	precision, recall, F = 0, 0, 0;
	for i in range(len(y_pred)):
		if y_actual[i]==y_pred[i]==1:
			TP += 1;
	for i in range(len(y_pred)): 
		if y_actual[i]==1 and y_actual!=y_pred[i]:
			FP += 1;
	for i in range(len(y_pred)): 
		if y_actual[i]==y_pred[i]==0:
			TN += 1;
	for i in range(len(y_pred)):
		if y_actual[i]==0 and y_actual!=y_pred[i]:
			FN += 1;
			
	TP = TP / total_sample;
	FP = FP / total_sample;	
	TN = TN / total_sample;	
	FN = FN / total_sample;
	precision = TP / (FP + TP);
	recall = TP / (FN + TP);
	F = 2*precision*recall/(precision+recall);
return (TP, FP, TN, FN, precision, recall, F);

def dataset_balancer(X_tr, X_te, y_tr, y_te, t_tr, t_te):
	occ_idx = np.where(y_tr == 1.)[0];
	unocc_idx = np.where(y_tr == 0.)[0];	
	removed_idx = [];
	removed_idx_te = [];
	if (len(occ_idx) > len(unocc_idx)):
		indices = np.full(len(occ_idx), False, bool);
		num_removal = len(occ_idx) - len(unocc_idx);
		randices = np.random.choice(np.arange(indices.shape[0]), num_removal, replace = False);
		indices[randices] = True;
		removed_idx = occ_idx[randices];
	elif (len(unocc_idx) > len(occ_idx)):
		indices = np.full(len(unocc_idx), False, bool);
		num_removal = len(unocc_idx) - len(occ_idx);
		randices = np.random.choice(np.arange(indices.shape[0]), num_removal, replace = False);
		indices[randices] = True;
		removed_idx = unocc_idx[randices];		
	X_tr = np.delete(X_tr, removed_idx, 0);
	y_tr = np.delete(y_tr, removed_idx);
	t_tr = np.delete(t_tr, removed_idx);
	return X_tr, X_te, y_tr, y_te, t_tr, t_te;
	
def split_weekends(dates):
	weekends = [];
	weekdays = [];
	for day in dates:
		date = dt.strptime(day, '%d-%b-%Y')
		if ((date.strftime('%A')=='Saturday') or (date.strftime('%A')=='Sunday')):
			weekends.append(date.strftime('%Y-%m-%d'));
		else:
			weekdays.append(date.strftime('%Y-%m-%d'));
	return weekends, weekdays;

def split_features_timestamps(weekdays, weekends, all_features, occ_label, timestamps):
	weekends_features = pd.DataFrame(columns = all_features.columns);
	weekdays_features = pd.DataFrame(columns = all_features.columns);
	weekends_occ_label = [];
	weekdays_occ_label = [];
	weekends_timestamps = [];
	weekdays_timestamps = [];
	occ_label = occ_label.reset_index();
	occ_label = occ_label.drop(occ_label.columns[[0,1]],1);
	for idx, timestamp in enumerate(timestamps):
		if str(timestamp)[0:10] in weekdays:
			weekdays_features = weekdays_features.append(all_features.iloc[idx]);
			weekdays_occ_label.append(occ_label.iloc[idx].values);
			weekdays_timestamps.append(timestamp);
		elif str(timestamp)[0:10] in weekends:
			weekends_features = weekends_features.append(all_features.iloc[idx]);
			weekends_occ_label.append(occ_label.iloc[idx].values); 
			weekends_timestamps.append(timestamp);
	return weekdays_features, weekends_features, weekdays_occ_label, weekends_occ_label, weekdays_timestamps, weekends_timestamps;

def split_all(weekends, weekdays, all_features, occ_label, timestamps):
	wd_features, we_features, wd_occ_label, we_occ_label, wd_timestamps, we_timestamps = split_features_timestamps(weekends, weekdays, all_features, occ_label, timestamps);
	wd_timestamps = np.array(wd_timestamps);
	we_timestamps = np.array(we_timestamps);
	wd_occ_label = np.array(wd_occ_label);
	we_occ_label = np.array(we_occ_label);
	sss_wd = StratifiedShuffleSplit(wd_occ_label, 1, test_size=test_ratio, random_state=0);
	for train_index, test_index in sss_wd:
		wd_timestamps_train, wd_timestamps_test = wd_timestamps[train_index], wd_timestamps[test_index];
		wd_X_train, wd_X_test = wd_features.as_matrix()[train_index], wd_features.as_matrix()[test_index];
		wd_y_train, wd_y_test = np.array(wd_occ_label)[train_index], np.array(wd_occ_label)[test_index];
	sss_we = StratifiedShuffleSplit(we_occ_label, 1, test_size=test_ratio, random_state=0);	
	for train_index, test_index in sss_we:
		we_timestamps_train, we_timestamps_test = we_timestamps[train_index], we_timestamps[test_index];
		we_X_train, we_X_test = we_features.as_matrix()[train_index], we_features.as_matrix()[test_index];
		we_y_train, we_y_test = np.array(we_occ_label)[train_index], np.array(we_occ_label)[test_index];
	timestamps_train = np.concatenate([wd_timestamps_train, we_timestamps_train]);
	timestamps_test = np.concatenate([wd_timestamps_test, we_timestamps_test]);
	X_train = np.concatenate([wd_X_train, we_X_train]);
	X_test = np.concatenate([wd_X_test, we_X_test]);
	y_train = np.concatenate([wd_y_train, we_y_train]);
	y_test = np.concatenate([wd_y_test, we_y_test]); 
	return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test;
	
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
	num_onoff = onoff.sum();
	return num_onoff;

def compute_feature(slot_data):
    slot_data = slot_data[slot_data>0].dropna();	
    if slot_data.empty:
        empty_feature = np.zeros(32);
        empty_feature = np.append(empty_feature, 'True');
        return empty_feature;
    min = slot_data.min();
    max = slot_data.max();
    mean = slot_data.mean();
    std = slot_data.std();
    sad = calculate_sad(slot_data);
    corl1 = slot_data.ix[:,0].autocorr();
    corl2 = slot_data.ix[:,1].autocorr();
    corl3 = slot_data.ix[:,2].autocorr();
    corl123 = slot_data.ix[:,3].autocorr();
    onoff = calculate_onoff(slot_data);
    range = max - min;
    isempty = 'False';
    feature = [min[0], min[1], min[2], min[3], max[0], max[1], max[2], max[3], mean[0], mean[1], mean[2], mean[3], std[0], std[1], std[2], std[3], sad[0], sad[1], sad[2], sad[3], corl1, corl2, corl3, corl123, onoff[0], onoff[1], onoff[2], onoff[3], range[0], range[1], range[2], range[3], isempty];
    return feature;

def compute_pprob(occ_data):
	occ_prob = occ_data[occ_data.columns[0:max_sampling_idx]].sum(axis=1).to_frame();
	for it in range(1, timeslot):
		idx = it * max_sampling_idx;
		occ_sum = occ_data[occ_data.columns[idx:idx+max_sampling_idx]].sum(axis=1).to_frame();
		occ_prob = pd.concat([occ_prob, occ_sum], axis=1);
	occ_prob = occ_prob.div(max_sampling_idx);
	occ_prob = occ_prob.stack();
	return occ_prob;

# extract features from raw_data
def extract_features(raw_data, occ_data, dates):
	a_features, ptime, pfixed, ptimes, pfixeds = [], [], [], [], [];
	slot_depart = 3 * (3600 / feature_length); # 9AM (6 + 3)
	slot_arrive = 11 * (3600 / feature_length); # 5PM (6 + 11)
	for slot in range(0, timeslot):
		ptime.append(slot+1);
		if slot >= slot_depart and slot <= slot_arrive:
			pfixed.append(1);
		else:
			pfixed.append(0);
	for day in range(0, len(dates)):
		ptimes.append(ptime);
		pfixeds.append(pfixed);
	ptimes = np.asarray(ptimes).flatten();
	pfixeds = np.asarray(pfixeds).flatten();
	pprob = compute_pprob(occ_data).to_frame();
	pprob = pprob.reset_index();
	dropped_cols = [0,1];
	pprob = pprob.drop(pprob.columns[dropped_cols], 1);
	timestamps = [];
	for day in dates:
		date = dt.strptime(day, '%d-%b-%Y');
		datestamp = date.strftime('%m/%d/%Y');
		timestamp = pd.date_range(datestamp, periods=timeslot, freq=feature_freq).values;
		timestamps.append(timestamp);			
	timestamps = np.asarray(timestamps).flatten().flatten();
	raw_data = raw_data.reset_index().drop('index', axis=1);
	start_time = time.time();	
	raw_data[3] = raw_data.sum(axis=1);
	d_features = raw_data.groupby(raw_data.index/max_sampling_idx).apply(compute_feature); # dataframe of 1 column containing an array of 32 feature values
	d_features = d_features.apply(lambda x:pd.Series(np.asarray(x).flatten())); # dataframe of 32 columns, each contains feature value
	print("--- compute_feature: %s seconds ---" % (time.time() - start_time));
	d_features.columns = ['min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'isempty'];
	d_features = d_features.set_index(timestamps).tshift(6, freq='H');
	d_features['ptime'] = pd.Series(ptimes, index=d_features.index);
	d_features['pfixed'] = pd.Series(pfixeds, index=d_features.index);
	d_features['pprob'] = pd.Series(pprob.ix[:,0].tolist(), index=d_features.index);
	timestamp = d_features.index.values;
	# normalize
	norm_d_features = d_features;
	norm_d_features = norm_d_features.drop('isempty', axis=1);
	x = norm_d_features.values; # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler();
	x_scaled = min_max_scaler.fit_transform(x); # sometimes contains NaN
	total_features = pd.DataFrame(x_scaled);
	total_features.columns = ['min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'pfixed', 'ptime', 'pprob'];
	total_features['isempty'] = d_features.reset_index()['isempty'];
	return total_features, timestamp;
	
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
	sm_path = '../../dataset/01_sm_csv/01_cross/';
	occ_path = '../../dataset/01_occupancy_csv/';
	occ_file = '01_summer.csv';
elif house=='r2':
	sm_path = '../../dataset/02_sm_csv/02_cross_weekends/';
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
weekends, weekdays = split_weekends(dates);
#print("--- load training data: %s seconds ---" % (time.time() - start_time));

# create ground truth data
start_time = time.time();
occ_data = read_occupancy(occ_file, dates);
total_occ_label = label_occupancy(occ_data);
#print("--- load occ_training_label: %s seconds ---" % (time.time() - start_time));

# extract features
start_time = time.time();
total_all_features, total_timestamps = extract_features(a_data, occ_data, dates);
filt_idx = total_all_features[total_all_features['isempty']=='False'].index;
all_features = total_all_features.iloc[filt_idx];
all_features = all_features.drop('isempty', axis=1);
print all_features.shape;
timestamps = np.array(total_timestamps)[filt_idx];
occ_label = np.array(total_occ_label)[filt_idx];
#print("--- extract all_features: %s seconds ---" % (time.time() - start_time));

# stratified ss with split weekdays and weekends
# X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = split_all(weekends, weekdays, all_features, occ_label, timestamps);

# cross validation
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_features, occ_label, test_size=test_ratio, random_state=0);
# timestamps_train = pd.DatetimeIndex(data=list(np.array(timestamps)[X_train.index]));
# timestamps_test = pd.DatetimeIndex(data=list(np.array(timestamps)[X_test.index]));

# stratified ss
sss = StratifiedShuffleSplit(occ_label, 1, test_size=test_ratio, random_state=0);
total_tp = [];
total_fp = [];
total_tn = [];
total_fn = [];
total_accuracies = [];
total_precision = [];
total_recall = []; 
total_F = [];
# kf = KFold(all_features.shape[0], shuffle=True, n_folds=2);
for train_index, test_index in sss:
	timestamps_train, timestamps_test = timestamps[train_index], timestamps[test_index];
	X_train, X_test = all_features.as_matrix()[train_index], all_features.as_matrix()[test_index];
	y_train, y_test = occ_label[train_index], occ_label[test_index];
	# X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = dataset_balancer(X_train, X_test, y_train, y_test, timestamps_train, timestamps_test);
	
	# load the features into pca	
	pca = PCA();
	pca.fit(X_train);
	#print("--- load training data to PCA: %s seconds ---" % (time.time() - start_time));

	# take only L components that make up the 95% variance
	# start_time = time.time();
	num_comp = 0;
	comp_sum = 0.0;
	ev_sum = np.sum(pca.explained_variance_ratio_);
	for ev in pca.explained_variance_ratio_:
		num_comp = num_comp+1;
		comp_sum = comp_sum+ev;
		if ((comp_sum / float(ev_sum)) > 0.95):
			break;
	#print("--- pca.n_components: %s ---" % num_comp);
	#print("--- find 0.95 variance: %s seconds ---" % (time.time() - start_time));

	# run PCA
	pca.n_components = num_comp;
	# print("PCA n components: %s" % num_comp);
	all_features_reduced = pca.fit_transform(X_train);
	# print("--- run PCA for training: %s seconds ---" % (time.time() - start_time));

	# run SVM classifier
	svc = svm.SVC(kernel='rbf', C=0.1, gamma=0.02);
	start_time = time.time();
	svc.fit(all_features_reduced, y_train);

	# c_params = np.arange(0.1,10,0.1);
	# gamma_params = np.arange(0.01,1,0.01);
	# params = {"C":c_params, "gamma": gamma_params};
	# grid_search = GridSearchCV(svc, params);
	# grid_search.fit(all_features_reduced, y_train);
	# print "grid_search best estimator: ";
	# print grid_search.best_estimator_;

	## TESTING PHASE
	# start_time = time.time();
	test_features_reduced = pca.fit_transform(X_test);
	# print("--- run PCA for testing: %s seconds ---" % (time.time() - start_time));

	# start_time = time.time();
	prediction = svc.predict(test_features_reduced);
	prediction_df = pd.DataFrame(data=prediction, index=timestamps_test);
	y_test_df = pd.DataFrame(data=y_test, index=timestamps_test);

	# group prediction by date
	mispred_df = abs(prediction_df - y_test_df);
	mispred_grouped = mispred_df.groupby(mispred_df.index.map(lambda x:str(x)[0:10])).mean();
	accuracy_grouped = mispred_grouped.apply(lambda x: 1-x);
	# accuracy_grouped.to_csv("accuracy_grouped.csv")
	accuracy = accuracy_grouped.mean()[0];
	total_accuracies.append(accuracy);
	
	TP, FP, TN, FN, precision, recall, F = perf_measure(y_test, prediction);
	total_tp.append(TP);
	total_fp.append(FP);
	total_tn.append(TN);
	total_fn.append(FN);
	total_precision.append(precision);
	total_recall.append(recall);
	total_F.append(F);
	
	perf_measure(prediction_df, y_test_df);	
	# print ("accuracy avg doubled: %s" % accuracy);

accuracy = np.array(total_accuracies).mean();
tp = np.array(total_tp).mean();
fp = np.array(total_fp).mean();
tn = np.array(total_tn).mean();
fn = np.array(total_fn).mean();
precision = np.array(total_precision).mean();
recall = np.array(total_recall).mean();
F = np.array(total_F).mean();

result = house + "," + str(test_ratio) + "," + str(sampling_rate) + "," + str(feature_length) + "," + str(accuracy);
with open("result_weekends_1.csv", "a") as myfile:
    myfile.write("\n");
    myfile.write(result);