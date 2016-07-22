import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
import os
import time

# Constants
START_IDX = 21600;
END_IDX = 82800;

# load data from CSV files inside the directory
def load_data(dir_path):
	a_data = []; # alltime data; d_da ta is daily data
	for i in os.listdir(dir_path):
		if i.endswith(".csv"):
			# read the data from 6 AM to 10 PM (data 21600 to 82800)
			d_data = pd.read_csv(filepath_or_buffer = dir_path + i, header=None, sep=',');
			a_data.append(d_data[START_IDX:END_IDX:1]);
	return a_data;

def compute_feature(slot, slot_data):
    min1 = slot_data.iloc[:,[0]].min().values[0]; 
    min2 = slot_data.iloc[:,[1]].min().values[0];
    min3 = slot_data.iloc[:,[2]].min().values[0];
    max1 = slot_data.iloc[:,[0]].max().values[0];
    max2 = slot_data.iloc[:,[1]].max().values[0];
    max3 = slot_data.iloc[:,[2]].max().values[0];
    mean1 = slot_data.iloc[:,[0]].mean().values[0];
    mean2 = slot_data.iloc[:,[1]].mean().values[0];
    mean3 = slot_data.iloc[:,[2]].mean().values[0];
    std1 = slot_data.iloc[:,[0]].std().values[0];
    std2 = slot_data.iloc[:,[1]].std().values[0];
    std3 = slot_data.iloc[:,[2]].std().values[0];
    sad1 = slot_data.iloc[:,[0]].abs().sum().values[0];
    sad2 = slot_data.iloc[:,[1]].abs().sum().values[0];
    sad3 = slot_data.iloc[:,[2]].abs().sum().values[0];
    range1 = max1 - min1;
    range2 = max2 - min2;
    range3 = max3 - min3;
    ptime = slot;
    if slot >= 12 and slot <=47 :
        pfixed = 1;
    else:
        pfixed = 0;
    feature = [min1, min2, min3, max1, max2, max3, mean1, mean2, mean3, std1, std2, std3, sad1, sad2, sad3, range1, range2, range3, pfixed, ptime];
    return feature;

# extract features from raw_data
def extract_features(raw_data):
	a_features = [];
	for day in raw_data:
		d_features = pd.DataFrame(columns=('min1', 'min2', 'min3','max1', 'max2', 'max3', 'mean1', 'mean2', 'mean3', 'std1', 'std2', 'std3', 'sad1', 'sad2', 'sad3', 'range1', 'range2', 'range3', 'pfixed', 'ptime'));
		# iterate over 17 * 4 slots = 68 slots (17 from 6AM to 10PM, 4 from 15 mins interval)
		for slot in range(0, 68):
			idx = slot * 900;
			d_features.loc[slot] = compute_feature(slot, day[idx:idx+900:1]);
		a_features.append(d_features);
	return pd.concat(a_features);

# find average occupancy for every 900 occupancy samples (occupancy per 15 minutes)
def label_occupancy(START_IDX_OCCUPANCY, END_IDX_OCCUPANCY):
	occ_data = pd.read_csv(filepath_or_buffer='../../dataset/02_occupancy_csv/02_summer.csv', skiprows=0, sep=',');
	occ_data = occ_data[START_IDX_OCCUPANCY:END_IDX_OCCUPANCY:1];
	occ_data = occ_data.drop(occ_data.columns[0:START_IDX+1], axis=1);
	occ_data = occ_data.drop(occ_data.columns[END_IDX-START_IDX:], axis=1);

	occ_label = occ_data[occ_data.columns[0:900]].mean(axis=1).to_frame();
	occ_label.columns = ['6'];
	for it in range(1, 68):
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

start_time = time.time();
all_features = extract_features(a_data);
print("--- extract all_features: %s seconds ---" % (time.time() - start_time));

# load the features into pca	
start_time = time.time();
pca = PCA();
pca.fit(all_features);
num_comp = 0;
comp_sum = 0.0;
ev_sum = np.sum(pca.explained_variance_);
print("--- load training data to PCA: %s seconds ---" % (time.time() - start_time));

# take only L components that make up the 95% variance
start_time = time.time();
for ev in pca.explained_variance_:
	num_comp = num_comp+1;
	comp_sum = comp_sum+ev;
	if ((comp_sum / ev_sum) > 0.95):
		break;
print("--- find 0.95 variance: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
pca.n_components = num_comp;
all_features_reduced = pca.fit_transform(all_features);
print("--- run PCA for training: %s seconds ---" % (time.time() - start_time));

# load to SVM
svc = svm.SVC(kernel='linear');

# create ground truth data for training, from 1 June (data #2) to 7 June (data #9)
start_time = time.time();
occ_label = label_occupancy(2, 9);
print("--- create training ground truth: %s seconds ---" % (time.time() - start_time));

# load occupancy ground truth
start_time = time.time();
svc.fit(all_features_reduced, occ_label);
print("--- run SVC: %s seconds ---" % (time.time() - start_time));

## TESTING PHASE
start_time = time.time();
test_data = load_data('../../dataset/02_sm_csv/02_test/');
print("--- load testing data: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
test_features = extract_features(test_data);
print("--- extract test_features: %s seconds ---" % (time.time() - start_time));

# create ground truth data for training, from 8 June (data #9) to 24 June (data #26Z)
start_time = time.time();
occ_test_label = label_occupancy(9, 26);
print("--- create occ_test_label: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
test_features_reduced = pca.fit_transform(test_features);
print("--- run PCA for testing: %s seconds ---" % (time.time() - start_time));

start_time = time.time();
prediction = svc.predict(test_features_reduced);
print("--- prediction: %s seconds ---" % (time.time() - start_time));
print accuracy_score(occ_test_label.values, prediction);
