import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
import os

# returns array containing computed features
def compute_feature(slot_data):
	min1 = slot_data.iloc[:,[0]].min().values[0]; 
	min2 = slot_data.iloc[:,[1]].min().values[0];
	min3 = slot_data.iloc[:,[2]].min().values[0];
	max1 = slot_data.iloc[:,[0]].max().values[0];
	max2 = slot_data.iloc[:,[1]].max().values[0];
	max3 = slot_data.iloc[:,[2]].max().values[0];
	feature = [min1, min2, min3, max1, max2, max3];
	return feature;

## TRAINING PHASE
# iterate over files and read the data from 6 AM to 10 PM (data 21600 to 82800)
START_IDX = 21600;
END_IDX = 82800;
START_IDX_OCCUPANCY = 2; # 1 June
END_IDX_OCCUPANCY = 5; # 3 June
a_data = []; # alltime data; d_data is daily data
for i in os.listdir('../../dataset/02_sm_csv/02_test/'):
    if i.endswith(".csv"): 
        d_data = pd.read_csv(filepath_or_buffer='../../dataset/02_sm_csv/02_test/' + i, header=None, sep=',');
        a_data.append(d_data[START_IDX:END_IDX:1]);

# extract features 
a_features = [];
for day in a_data:
	d_features = pd.DataFrame(columns=('min1', 'min2', 'min3','max1', 'max2', 'max3'));
	# iterate over 17 * 4 slots = 68 slots (17 from 6AM to 10PM, 4 from 15 mins interval)
	for slot in range(0, 68): 
		idx = slot * 900;
		d_features.loc[slot] = compute_feature(day[idx:idx+900:1]);
	a_features.append(d_features);

# load the features into pca	
all_features = pd.concat(a_features);
pca = PCA();
pca.fit(all_features);
num_comp = 0;
comp_sum = 0.0;
ev_sum = np.sum(pca.explained_variance_);

# take only L components that make up the 95% variance
for ev in pca.explained_variance_:
	num_comp = num_comp+1;
	comp_sum = comp_sum+ev;
	if ((comp_sum / ev_sum) > 0.95):
		break;

pca.n_components = num_comp;
all_features_reduced = pca.fit_transform(all_features);

# load to SVM
svc = svm.SVC(kernel='linear');
# create ground truth data
occ_data = pd.read_csv(filepath_or_buffer='../../dataset/02_occupancy_csv/02_summer.csv', skiprows=0, sep=',');
occ_data = occ_data[START_IDX_OCCUPANCY:END_IDX_OCCUPANCY:1];
occ_data = occ_data.drop(occ_data.columns[0:START_IDX+1], axis=1);
occ_data = occ_data.drop(occ_data.columns[END_IDX-START_IDX:], axis=1);

# find average occupancy for every 900 occupancy samples (occupancy per 15 minutes)
occ_label = occ_data[occ_data.columns[0:900]].mean(axis=1).to_frame();
occ_label.columns = ['6'];
for it in range(1, 68):
	idx = it * 900;
	occ_mean = occ_data[occ_data.columns[idx:idx+900]].mean(axis=1).to_frame();
	occ_mean.columns = [str(it + 6)];
	occ_label = pd.concat([occ_label, occ_mean], axis=1);

occ_label = occ_label.round();
occ_label = occ_label.stack();

# load occupancy ground truth
svc.fit(all_features_reduced, occ_label);

## TESTING PHASE
prediction = svc.predict(all_features_reduced);
print accuracy_score(occ_label.values, prediction);
