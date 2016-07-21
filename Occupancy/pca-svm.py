import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#import statistics as st
import os

# returns array containing computed features
print("function compute feature run..")
def compute_feature( slot_ , slot_data ):
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
    feature = [min1, min2, min3, max1, max2, max3, mean1, mean2, mean3, std1, std2, std3, sad1, sad2, sad3, range1, range2, range3];
    return feature;

## TRAINING PHASE
# iterate over files and read the data from 6 AM to 10 PM (data 21600 to 82800)
print("fetching data run..")
START_IDX = 21600;
END_IDX = 82800;
a_data = []; # alltime data; d_data is daily data
for i in os.listdir('../dataset/02_sm_csv/02_test/'):
    if i.endswith(".csv"): 
        d_data = pd.read_csv(filepath_or_buffer='../dataset/02_sm_csv/02_test/' + i, header=None, sep=',');
        a_data.append(d_data[START_IDX:END_IDX:1]);
        #Now a_data cotains everyday data from 21600 to 82800

# extract features 
print("feature extraction..")
a_features = [];
for day in a_data:
    d_features = pd.DataFrame(columns=('min1', 'min2', 'min3', 'min123', 'max1', 'max2', 'max3', 'max123', 'mean1','mean2','mean3', 'std1', 'std2', 'std3', 'sad1', 'sad2', 'sad3', 'range1', 'range2', 'range3', 'pfixed'));
    for slot in range(0, 68):
        # iterate over 17 * 4 slots = 68 slots (17 from 6AM to 10PM, 4 from 15 mins interval--> 1 slot = 15 min data)
        idx = slot * 900; #900 seconds = 15 min
        d_features.loc[slot] = compute_feature(slot_, day[idx:idx+900:1]); #looking for min max values for every 900 seconds data (each slot)
    a_features.append(d_features);
#print("a_features")
#print(a_features)

# load the features into pca
print("load feature to PCA module..")
all_features = pd.concat(a_features);
print("concatenated a_features")
print(all_features)
pca = PCA();
pca.fit(all_features);
num_comp = 0;
comp_sum = 0.0;
ev_sum = np.sum(pca.explained_variance_);

#print("print explained variance");
#print (pca.explained_variance_);
#print("print ev_sum");
#print (ev_sum);

# take only L components that make up the 95% variance
for ev in pca.explained_variance_:
	num_comp = num_comp+1;
	comp_sum = comp_sum+ev;
	if ((comp_sum / ev_sum) > 0.95):
		break;

pca.n_components = num_comp;
all_features_reduced = pca.fit_transform(all_features);

#debug purposed
#print(all_features_reduced);
# load to SVM

## TESTING PHASE