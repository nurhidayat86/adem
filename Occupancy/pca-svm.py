import pandas as pd
from sklearn.decomposition import PCA
import os

def compute_feature(slot_data):
	return 0;

## TRAINING PHASE
# iterate over files and read the data from 6 AM to 10 PM (data 21600 to 82799)
START_IDX = 21600;
END_IDX = 82800;
alltime_data = [];
for i in os.listdir('../../dataset/02_sm_csv/02_test/'):
    if i.endswith(".csv"): 
        daily_data = pd.read_csv(filepath_or_buffer='../../dataset/02_sm_csv/02_test/' + i, header=None, sep=',');
        alltime_data.append(daily_data[START_IDX:END_IDX:1]);

# extract features 
for day in alltime_data:
	# iterate over 17 * 4 slots = 68 slots (17 from 6AM to 10PM, 4 from 15 mins interval)
	for slot in range(0, 68): 
		idx = slot * 900;
		feature = compute_feature(day[idx:idx+900:1]);

		
# load into pca

# load to SVM

## TESTING PHASE