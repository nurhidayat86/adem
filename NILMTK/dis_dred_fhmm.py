from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
#%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

from nilmtk.electric import align_two_meters
from nilmtk.metergroup import iterate_through_submeters_of_two_metergroups    
import numpy as np
import pandas as pd
import math

def FTE_func(predictions, ground_truth):
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)   
    total_pred_list = []
    total_gt_list = []
    fraction_pred = []
    fraction_gt = []
    fraction_min = []
    total_pred = 0.0  
    total_gt = 0.0
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        total_app_pred = 0.0
        total_app_gt = 0.0
        for aligned_meters_chunk in align_two_meters(pred_meter, ground_truth_meter):
            total_pred += aligned_meters_chunk.icol(0).sum()
            total_gt += aligned_meters_chunk.icol(1).sum()
            total_app_pred += aligned_meters_chunk.icol(0).sum()
            total_app_gt += aligned_meters_chunk.icol(1).sum()
        total_pred_list.append(total_app_pred)   
        total_gt_list.append(total_app_gt)  
    fraction_gt = np.array(total_gt_list)/total_gt
    fraction_pred = np.array(total_pred_list)/total_pred
    for i in range(len(fraction_pred)):
        fraction_min.append(min(fraction_pred[i], fraction_gt[i])) 
    return np.array(fraction_min).sum()

def fraction_energy_assigned_correctly_nan(predictions, ground_truth):
    predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)
    fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth.index = fraction_per_meter_ground_truth.index.map(lambda meter: meter.instance)
    fraction_per_meter_predictions.index = fraction_per_meter_predictions.index.map(lambda meter: meter.instance)
    fraction = 0
    for meter_instance in predictions_submeters.instance():
        if math.isnan(fraction_per_meter_ground_truth[meter_instance]) == False:
            fraction += min(fraction_per_meter_ground_truth[meter_instance],
                        fraction_per_meter_predictions[meter_instance])
    return fraction

# based on method : def mean_normalized_error_power(predictions, ground_truth):
def total_disag_err(predictions, ground_truth):
    #only iterate for the instance in the prediction/ elecmeter with lesser instance
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)   
    # additional of total variable
    total_diff = 0.0
    total_pred = 0.0  
    total_gt = 0.0 
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        for aligned_meters_chunk in align_two_meters(pred_meter, ground_truth_meter):
            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
            total_pred += aligned_meters_chunk.icol(0).sum()
            total_diff += sum(abs(diff.dropna()))
	    total_gt += aligned_meters_chunk.icol(1).sum()
    return float(total_diff)/float(total_gt)

def jaccard_similarity(predictions, ground_truth, submeters_instance):
    predictions = predictions.drop(predictions.columns[[0]], axis=1)
    ind = submeters_instance 
    ind = list(map(lambda x: x - 1, ind))
    ind_test = test_elec.instance() 
    ind_test = list(map(lambda x: x - 1, ind_test))
    ind_drop = list(set(ind_test) - set(ind))
    ground_truth = ground_truth.drop(ground_truth.columns[ind_drop], axis=1)
    predictions.columns = [ind]
    ground_truth.columns = [ind]
    pred_a_gt = predictions.astype(bool) & ground_truth.astype(bool)
    pred_u_gt = predictions.astype(bool) | ground_truth.astype(bool)
    sum_pred_a_gt = pred_a_gt.sum(axis=1)
    sum_pred_u_gt = pred_u_gt.sum(axis=1)
    ratio = sum_pred_a_gt / sum_pred_u_gt
    Ja = ratio.sum() / len(ratio)
    return Ja

### FHMM METRIC CALCULATION ###
period_s = 60
dataset_loc = '../../dataset/DRED.h5';

#load train and test data
total = DataSet(dataset_loc)
train = DataSet(dataset_loc)
test = DataSet(dataset_loc)

# train and test visualization
#train.buildings[1].elec.mains().plot()
#test.buildings[1].elec.mains().plot()t
#plt.show()

# change this variable to get the result from different building
building = 1

### TimeFrame(start='2015-07-05 00:00:08+02:00', end='2015-12-05 12:00:01+01:00', empty=False)
tf_total = total.buildings[building].elec.mains().get_timeframe()

# 5 7 2015 - 31 8 2015
# 9 1 2015 - 9 9 2015

#train.set_window(start="07-05-2015", end="08-05-2015")
#train.set_window(start="10-05-2015", end="11-05-2015")
#test.set_window(start="10-05-2015", end="10-06-2015")
 
train.set_window(start="07-05-2015", end="08-31-2015")
test.set_window(start="09-01-2015", end="09-09-2015")
   
total_elec = total.buildings[building].elec
train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

tf_train = train.buildings[building].elec.mains().get_timeframe()
tf_test = test.buildings[building].elec.mains().get_timeframe()

tf_train
tf_test

# select top k submeter
top_5_train_elec = train_elec.submeters().select_top_k(k=5)

"""
# CO disaggregation test
start = time.time()
from nilmtk.disaggregate import CombinatorialOptimisation
co = CombinatorialOptimisation()
# Note that we have given the sample period to downsample the data to 15 minute
#co.train(top_5_train_elec, sample_period=period_s)
co.train(train_elec, sample_period=period_s)
#co.train(train_elec[5], sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")
"""

# FHMM
start = time.time()
from nilmtk.disaggregate import fhmm_exact
co = fhmm_exact.FHMM()
# Note that we have given the sample period to downsample the data to 1 minute
co.train(train_elec, sample_period=period_s)
#fhmm.train(train_elec.submeters(), sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")

### PRINT MODEL RESULT ####

"""
for i, model in enumerate(co.model):
    print(model['training_metadata'].instance(), train_elec[model['training_metadata'].instance()].label(), model['states'])
"""

start = time.time()
disag_filename = '../../DRED_test.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 900 seconds
co.disaggregate(test_elec.mains(), output, sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")
output.close()

disag_co = DataSet(disag_filename)
disag_co_elec = disag_co.buildings[building].elec

# FTE
FTE_co = fraction_energy_assigned_correctly_nan(disag_co_elec, test_elec)
FTE_co2 = FTE_func(disag_co_elec, test_elec)

# Te
Te_co = total_disag_err(disag_co_elec, test_elec)

###############################################################################################
# Jaccard (Number of Appliances Identified Correctly ##########################################
###############################################################################################

# creating dataframe from both disaggregated and ground truth metergroups
disag_co_elec_df = disag_co_elec.dataframe_of_meters();
disag_co_elec_df_nona = disag_co_elec_df.dropna();
gt_full_df = test_elec.dataframe_of_meters();
gt_full_df_nona = gt_full_df.dropna();
gt_df_nona = gt_full_df_nona.ix[disag_co_elec_df_nona.index];

# Ja
Ja_co = jaccard_similarity(disag_co_elec_df_nona, gt_df_nona, disag_co_elec.submeters().instance());

print(FTE_co)
print(FTE_co2)
print(Te_co)
print(Ja_co)

for i in disag_co_elec.submeters().instance():
    plt.clf()
    train_elec[i].plot()
    #disag_co_elec[i].plot()
    fig_name = test_elec[i].label()
    plt.savefig('../../train/' + fig_name + '.png')

for i in train_elec.submeters().instance():
    plt.clf()
    train_elec[i].plot()
    #disag_co_elec[i].plot()
    fig_name = test_elec[i].label()
    plt.savefig('../../train/' + fig_name + '.png')


plt.clf()
from nilmtk.metrics import f1_score
f1_co= f1_score(disag_co_elec, test_elec)
f1_co.index = disag_co_elec.get_labels(f1_co.index)
f1_co.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("CO");
plt.savefig('f1.png')

"""
for i in test_elec.submeters().instance():
    plt.clf()
    test_elec[i].plot()
    fig_name = test_elec[i].label()
    plt.savefig('/media/airawan/DATA/Data/ecoori/test' + fig_name + ' 2d_eco.png')

for i, model in enumerate(co.model):
    print(model['training_metadata'].instance(), train_elec[model['training_metadata'].instance()].label(), model['states'])
"""

"""

# CO disaggregation k all
start = time.time()
from nilmtk.disaggregate import CombinatorialOptimisation
co = CombinatorialOptimisation()
# Note that we have given the sample period to downsample the data to 15 minute
co.train(train_elec.submeters(), sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")

start = time.time()
disag_filename = '/media/airawan/DATA/Data/eco-b2-kall-co-1:1-1d1m.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 900 seconds
co.disaggregate(test_elec.mains(), output, sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")
output.close()

disag_co = DataSet(disag_filename)
disag_co_elec = disag_co.buildings[building].elec

# FTE
FTE_co = fraction_energy_assigned_correctly_nan(disag_co_elec, test_elec)

# Te
Te_co = total_disag_err(disag_co_elec, test_elec)

# creating dataframe from both disaggregated and ground truth metergroups
disag_co_elec_df = disag_co_elec.dataframe_of_meters();
disag_co_elec_df_nona = disag_co_elec_df.dropna();
gt_full_df = test_elec.dataframe_of_meters();
gt_full_df_nona = gt_full_df.dropna();
gt_df_nona = gt_full_df_nona.ix[disag_co_elec_df_nona.index];

# Ja
Ja_co = jaccard_similarity(disag_co_elec_df_nona, gt_df_nona);
print(FTE_co)
print(Te_co)
print(Ja_co);
"""

"""
# f1_score
from nilmtk.metrics import f1_score
f1_co= f1_score(disag_co_elec, test_elec)
f1_co.index = disag_co_elec.get_labels(f1_co.index)
f1_co.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("CO");
"""

"""
# MNe
MNe_co = mean_normalized_error_power(disag_co_elec, test_elec)
MNe_co.index = disag_co_elec.get_labels(MNe_co.index)
Mne_co.plot(kind='barh')
MNe_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('Te');
plt.title("CO and FHMM");
plt.show()
"""

"""
# FHMM
start = time.time()
from nilmtk.disaggregate import fhmm_exact
fhmm = fhmm_exact.FHMM()
# Note that we have given the sample period to downsample the data to 1 minute
fhmm.train(top_8_train_elec, sample_period=period_s)
#fhmm.train(train_elec.submeters(), sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")

start = time.time()
disag_filename = '/media/airawan/DATA/Data/eco-b2-k8-fhmm-0.75:1-15m.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
fhmm.disaggregate(test_elec.mains(), output, sample_period=period_s)
end = time.time()
print("Runtime =", end-start, "seconds.")
output.close()

disag_filename = '/media/airawan/DATA/Data/eco-b2-k8-fhmm-0.75:1-15m.h5'
disag_fhmm = DataSet(disag_filename)
disag_fhmm_elec = disag_fhmm.buildings[building].elec

### FHMM METRIC CALCULATION ###
"""

"""
# f1_score calculation
from nilmtk.metrics import f1_score
f1_fhmm = f1_score(disag_fhmm_elec, test_elec)
f1_fhmm.index = disag_fhmm_elec.get_labels(f1_fhmm.index)
f1_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("FHMM");
"""

"""
# mean_norm_error
from nilmtk.metrics import mean_normalized_error_power
MNe_fhmm = mean_normalized_error_power(disag_fhmm_elec, test_elec)
MNe_fhmm.index = disag_fhmm_elec.get_labels(MNe_fhmm.index)
MNe_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('MNerr');
plt.title("FHMM");
plt.show()

"""

"""
# metric FTE
from nilmtk.metrics import fraction_energy_assigned_correctly
FTE_fhmm = fraction_energy_assigned_correctly_nan(disag_fhmm_elec, test_elec)

# Te
Te_fhmm = total_disag_err(disag_fhmm_elec, test_elec)

# creating dataframe from both disaggregated and ground truth metergroups
disag_fhmm_elec_df = disag_fhmm_elec.dataframe_of_meters();
disag_fhmm_elec_df_nona = disag_fhmm_elec_df.dropna();
gt_full_df = test_elec.dataframe_of_meters();
gt_full_df_nona = gt_full_df.dropna();
gt_df_nona = gt_full_df_nona.ix[disag_fhmm_elec_df_nona.index];

# Ja
print("starting Ja co")
Ja_co = jaccard_similarity_b2_k8(disag_fhmm_elec_df_nona, gt_df_nona)
print(Ja_co)
"""

"""
# write disaggregation output format 1
target = open("output_format.csv", 'w')
size = disag_co_elec[1].load().next().axes[0].size
for i in range(size):
    data = ''
    data += str(disag_co_elec[instance].load().next().axes[0][i].value)
    data += ' '
    for instance in disag_co_elec.submeters().instance():
        if disag_co_elec[instance].load().next().ix[i][0] != 0:
            #print(disag_co_elec[instance].label())
            data += disag_co_elec[instance].label()
            data += ' '
            data += str(disag_co_elec[instance].load().next().ix[i][0])
            data += ', '        
    data += '\n'
    #print(data)
    target.write(data)

target.close()

# write disaggregation output format 2 (colomn)
target = open("output_7_actual.csv", 'w')

data = 'timestamp'
data += '\t'    
for instance in test_elec.submeters().instance():
    data += test_elec[instance].label()            
    data += '\t'
    #print(data)

data += '\r\n'
target.write(data)

size = test_elec[1].load().next().axes[0].size
for i in range(size):
    data = ''
    data += str(test_elec[instance].load().next().axes[0][i].value)
    data += '\t'
    for instance in test_elec.submeters().instance():
        data += str(test_elec[instance].load().next().ix[i][0])
        data += '\t'        
    
    data += '\r\n'
    #print(data)
    target.write(data)

target.close()

# write disaggregation output format 2 (colomn)
target = open("output_b3_15m.csv", 'w')

data = 'timestamp'
data += '\t'    
for instance in disag_co_elec.submeters().instance():
    data += disag_co_elec[instance].label()            
    data += '\t'
    #print(data)

data += '\r\n'
target.write(data)

size = disag_co_elec[1].load().next().axes[0].size
for i in range(size):
    data = ''
    data += str(disag_co_elec[instance].load().next().axes[0][i].value)
    data += '\t'
    for instance in disag_co_elec.submeters().instance():
        data += str(disag_co_elec[instance].load().next().ix[i][0])
        data += '\t'        
    
    data += '\r\n'
    #print(data)
    target.write(data)

target.close()


# check appliances state
target = open("output2.csv", 'w')

data = ''    
for instance in disag_fhmm_elec.submeters().instance():
    data += disag_fhmm_elec[instance].label()            
    data += '\t'
    #print(data)

data += '\r\n'
target.write(data)
    
n11 = 0.0
n10 = 0.0
n01 = 0.0
n00 = 0.0

size = disag_fhmm_elec[1].load().next().axes[0].size
for i in range(int(50)):
    #data = ''
    for instance in disag_fhmm_elec.submeters().instance():
        if disag_fhmm_elec.submeters()[instance].load().next().ix[i][0] != 0:
            if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
                #data += '1'
                n11 = n11+1
            else:
                n10 = n10+1       
	else:
            #data += '0'
            if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
                #data += '1'
                n01 = n01+1
            else:
                n00 = n00+1     
    
        #data += '\t'
    #data += '\r\n'
    #target.write(data)
    #break

target.close()

#compute Ja
n11 = 0.0
n10 = 0.0
n01 = 0.0
n00 = 0.0

size = disag_fhmm_elec[1].load().next().axes[0].size
for i in range(size):
    for instance in disag_fhmm_elec.submeters().instance():
        if disag_fhmm_elec.submeters()[instance].load().next().ix[i][0] != 0:
            if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
                n11 = n11+1
            else:
                n10 = n10+1       
	else:
            if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
                n01 = n01+1
            else:
                n00 = n00+1     
    
target.close()

Ja = (n11+n00)/(n10+n11+n01+n00)

"""
