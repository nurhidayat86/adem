import pandas as pd
from sklearn.decomposition import PCA
import os

# returns array containing computed features
def compute_feature(slot_data):
	return 0;

## TRAINING PHASE
# iterate over files and read the data from 6 AM to 10 PM (data 21600 to 82800)
START_IDX = 21600;
END_IDX = 82800;
a_data = []; # alltime data; d_data is daily data
for i in os.listdir('../../dataset/02_sm_csv/02_test/'):
    if i.endswith(".csv"): 
        d_data = pd.read_csv(filepath_or_buffer='../../dataset/02_sm_csv/02_test/' + i, header=None, sep=',');
        a_data.append(d_data[START_IDX:END_IDX:1]);

# extract features 
a_features = [];
for day in a_data:
	d_features = DataFrame(columns=('min1', 'min2', 'min3','min123','max1', 'max2', 'max3','max123'));
	# iterate over 17 * 4 slots = 68 slots (17 from 6AM to 10PM, 4 from 15 mins interval)
	for slot in range(0, 68): 
		idx = slot * 900;
		d_features.loc[slot] = compute_feature(day[idx:idx+900:1]);
	a_features.append(d_features);
		
# load into pca

# load to SVM

## TESTING PHASE