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

# Constants
START_IDX = 21600;
END_IDX = 80100;
DELTA = 20.0; # >30 watts indicates on off events

# load data from CSV files inside the directory
def load_data(dir_path):
	a_data = []; # alltime data; d_data is daily data
	for i in os.listdir(dir_path):
		if i.endswith(".csv"):
			# read the data from 6 AM to 10 PM (data 21600 to 80100)
			d_data = pd.read_csv(filepath_or_buffer = dir_path + i, header=None, sep=',', usecols=[0,1,2]);
			a_data.append(d_data[START_IDX:END_IDX:1]);
	return a_data;	
	
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

def compute_pprob(occ_data):
	occ_prob = occ_data[occ_data.columns[0:900]].sum(axis=1).to_frame();
	for it in range(1, 65):
		idx = it * 900;
		occ_sum = occ_data[occ_data.columns[idx:idx+900]].sum(axis=1).to_frame();
		occ_prob = pd.concat([occ_prob, occ_sum], axis=1);	
	occ_prob = occ_prob.div(900);
	occ_prob = occ_prob.stack();
	return occ_prob;

# extract features from raw_data
def extract_features(raw_data, occ_data):
	a_features = [];
	for day in raw_data:
		day = day[day > 0].dropna(); # remove data with negative power value		
		d_features = pd.DataFrame(columns=('min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'pfixed', 'ptime'));
		# iterate over 16 * 4 + 1` slots = 65 slots (16 from 6AM to 9PM, 4 from 15 mins interval, 10PM only contributes 1)
		for slot in range(0, 65):
			idx = slot * 900;
			d_features.loc[slot] = compute_feature(slot, day[idx:idx+900:1]);
		a_features.append(d_features);

	total_features = pd.concat(a_features);	
	total_features = total_features.reset_index();
	total_features = total_features.drop('index', 1);
	pprob = compute_pprob(occ_data).to_frame();
	pprob = pprob.reset_index();
	dropped_cols = [0,1];
	pprob = pprob.drop(pprob.columns[dropped_cols], 1);
	pprob.columns = ['pprob'];
	total_features = total_features.join(pprob);
	
	# normalize data. should be on the features, not on the data!
	x = total_features.values; # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler();
	x_scaled = min_max_scaler.fit_transform(x);
	total_features = pd.DataFrame(x_scaled);
	total_features.columns = ['min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1', 'mean2', 'mean3', 'mean123', 'std1', 'std2', 'std3', 'std123', 'sad1', 'sad2', 'sad3', 'sad123', 'corl1', 'corl2', 'corl3', 'corl123', 'onoff1', 'onoff2', 'onoff3', 'onoff123', 'range1', 'range2', 'range3', 'range123', 'pfixed', 'ptime', 'pprob'];
	return total_features;

def read_occupancy(occ_filename, START_IDX_OCCUPANCY, END_IDX_OCCUPANCY):	
	occ_data = pd.read_csv(filepath_or_buffer='../../dataset/02_occupancy_csv/' + occ_filename, skiprows=0, sep=',');
	occ_data = occ_data[START_IDX_OCCUPANCY:END_IDX_OCCUPANCY:1];
	occ_data = occ_data.drop(occ_data.columns[0:START_IDX+1], axis=1);
	occ_data = occ_data.drop(occ_data.columns[END_IDX-START_IDX:], axis=1);
	return occ_data;
	
# find average occupancy for every 900 occupancy samples (occupancy per 15 minutes)
def label_occupancy(occ_data):	
	occ_label = occ_data[occ_data.columns[0:900]].mean(axis=1).to_frame();
	occ_label.columns = ['6'];
	for it in range(1, 65):
		idx = it * 900;
		occ_mean = occ_data[occ_data.columns[idx:idx+900]].mean(axis=1).to_frame();
		occ_mean.columns = [str(it + 6)];
		occ_label = pd.concat([occ_label, occ_mean], axis=1);	
	occ_label = occ_label.round();
	occ_label = occ_label.stack();
	return occ_label;
	
## TRAINING PHASE
start_time = time.time();
a_data = load_data('../../dataset/02_sm_csv/02_train/');
print("--- load training data: %s seconds ---" % (time.time() - start_time));

# create ground truth data for training, from 1 June (data #2) to 7 June (data #9)
start_time = time.time();
occ_training_data = read_occupancy('02_summer.csv', 2, 9);
occ_training_label = label_occupancy(occ_training_data);
print("--- load occ_training_label: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
all_features = extract_features(a_data, occ_training_data);
f = open('all_features.csv', 'w');
f.write(all_features.to_csv());
f.close();
print("--- extract all_features: %s seconds ---" % (time.time() - start_time));

# load the features into pca	
start_time = time.time();
pca = PCA();
pca.fit(all_features);
print("--- load training data to PCA: %s seconds ---" % (time.time() - start_time));

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
print("--- pca.n_components: %s ---" % num_comp);
print("--- find 0.95 variance: %s seconds ---" % (time.time() - start_time));

# run PCA
start_time = time.time();
pca.n_components = num_comp;
all_features_reduced = pca.fit_transform(all_features);
print all_features_reduced.shape;
print("--- run PCA for training: %s seconds ---" % (time.time() - start_time));

# run SVM classifier
svc = svm.SVC(kernel='rbf', C=1.4000000000000001, gamma=0.050000000000000003);
start_time = time.time();
svc.fit(all_features_reduced, occ_training_label);

# c_params = np.arange(0.1,10,0.1);
# gamma_params = np.arange(0.001,1,0.001);
# params = {"C":c_params, "gamma": gamma_params};
# grid_search = GridSearchCV(svc, params);
# grid_search.fit(all_features_reduced, occ_training_label);
# print "grid_search best estimator: ";
# print grid_search.best_estimator_;
print("--- run SVC: %s seconds ---" % (time.time() - start_time));

## TESTING PHASE
start_time = time.time();
test_data = load_data('../../dataset/02_sm_csv/02_test/');
print("--- load testing data: %s seconds ---" % (time.time() - start_time));

# create ground truth data for testing, from 8 June (data #9) to 14 June (data #16)
start_time = time.time();
occ_test_data = read_occupancy('02_summer.csv', 9, 16);
occ_test_label = label_occupancy(occ_test_data);
print("--- load occ_test_label: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
test_features = extract_features(test_data, occ_test_data);
print("--- extract test_features: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
test_features_reduced = pca.fit_transform(test_features);
print("--- run PCA for testing: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
prediction = svc.predict(test_features_reduced);
print("--- prediction: %s seconds ---" % (time.time() - start_time));
print accuracy_score(occ_test_label.values, prediction);