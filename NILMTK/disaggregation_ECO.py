from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
#%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.metrics import fraction_energy_assigned_correctly

from nilmtk.electric import align_two_meters
from nilmtk.metergroup import iterate_through_submeters_of_two_metergroups    
import numpy as np
import pandas as pd

from math import sqrt
import nilmtk.metrics as nilmtk_metrics
from nilmtk.metergroup import MeterGroup
from pandas import DataFrame, Series
import cPickle as pickle

#load train and test data
total = DataSet('../../dataset/eco.h5')
train = DataSet('../../dataset/eco.h5')
test = DataSet('../../dataset/eco.h5')

# train and test visualization
#plt.show (train.buildings[1].elec.mains().plot())
#plt.show (test.buildings[1].elec.mains().plot())

# change this variable to get the result from different building
building = 2

# start 2012-06-01 end 2013-02-01
tf_total = total.buildings[building].elec.mains().get_timeframe()

# train and test interval
if building == 1:
    """
    #Train 2 Month, Test 4 Month
    train.set_window(start="2012-08-01",end="30-09-2012")
    test.set_window(start="2012-10-01",end="23-01-2013")
    
     #Train 3 Month, Test 4 Month
    train.set_window(start="2012-07-01",end="30-09-2012")
    test.set_window(start="2012-10-01",end="23-01-2013")
    """
    #Train 4 Month, Test 4 Month
    train.set_window(start="2012-06-01",end="30-09-2012")
    test.set_window(start="2012-10-01",end="23-01-2013")
    print(1)
    
elif building == 2:
    """
    #Train 1 Month, Test 7 Month
    train.set_window(start="2012-06-01",end="30-06-2012")
    test.set_window(start="2012-07-01",end="2013-02-01")
    
    #Train 2 Month, Test 6 Month
    train.set_window(start="2012-06-01",end="31-07-2012")
    test.set_window(start="2012-08-01",end="2013-02-01")
    
    #Train 3 Month, Test 5 Month
    train.set_window(start="2012-06-01",end="31-08-2012")
    test.set_window(start="2012-09-01",end="2013-02-01")
    """
    #Train 3 Month, Test 5 Month
    #train.set_window(start="06-01-2012",end="07-10-2012")
    #test.set_window(start="07-11-2012",end="09-01-2013")
    
	#Train 3/4 Month, Test 1 Month
    train.set_window(start="06-02-2012",end="06-08-2012")
    test.set_window(start="06-09-2012",end="06-20-2012")	
    print(2)
    
    
elif building == 3:
    """
    #Train 2 Month, Test 2 Month
    train.set_window(start="23-10-2012",end="30-11-2012")
    test.set_window(start="2012-12-01",end="2013-02-01")
    #Train 3 Month, Test 1 Month
    train.set_window(start="23-10-2012",end="31-12-2012")
    test.set_window(start="2013-01-01",end="2013-02-01")
    """
   #Train 3 Month, Test 1 Month
    train.set_window(start="23-10-2012",end="31-12-2012")
    test.set_window(start="2013-01-01",end="2013-02-01")
    
    print(3)

total_elec = total.buildings[building].elec
train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

tf_train = train.buildings[building].elec.mains().get_timeframe()
tf_test = test.buildings[building].elec.mains().get_timeframe()

tf_train
tf_test

print("finished tf_train tf_test");

#show the plot of train or test after splitting the  time 
#plt.show (train_elec.mains().plot())
#plt.show (test_elec.mains().plot())

# check data sampling time
#fridge_meter = train_elec['fridge', 1]
#fridge_df = fridge_meter.load().next()
#fridge_df.head()

# select top k submeter
top_8_train_elec = train_elec.submeters().select_top_k(k=8)
top_8_train_elec

###############################################################################################
# FHMM  #######################################################################################
###############################################################################################
# print("starting FHMM");
# start = time.time()
# from nilmtk.disaggregate import fhmm_exact
# fhmm = fhmm_exact.FHMM()
# # Note that we have given the sample period to downsample the data to 1 minute
# fhmm.train(top_8_train_elec, sample_period=60*15)
# #fhmm.train(train_elec.submeters(), sample_period=60)
# end = time.time()
# print("Runtime =", end-start, "seconds.")
# 
# start = time.time()
# disag_filename = 'data/fhmm_out/fhmm_building2_3vs5.h5'
# output = HDFDataStore(disag_filename, 'w')
# # Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
# fhmm.disaggregate(test_elec.mains(), output, sample_period=60*15)
# end = time.time()
# print("Runtime =", end-start, "seconds.")
# output.close()
# 
# disag_fhmm = DataSet(disag_filename)
# disag_fhmm_elec = disag_fhmm.buildings[building].elec
# print("finish FHMM");

###############################################################################################
# CO  #########################################################################################
###############################################################################################
print("starting CO");
start = time.time()
from nilmtk.disaggregate import CombinatorialOptimisation
co = CombinatorialOptimisation()
# Note that we have given the sample period to downsample the data to 1 minute
#co.train(top_8_train_elec, sample_period=60)
co.train(train_elec.submeters(), sample_period=60*15)
end = time.time()
print("Runtime =", end-start, "seconds.")

start = time.time()
disag_filename = 'data/co_out/co_building2ver2_2vs6.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
co.disaggregate(test_elec.mains(), output, sample_period=60*15)
end = time.time()
print("Runtime =", end-start, "seconds.")
output.close()

disag_co = DataSet(disag_filename)
disag_co_elec = disag_co.buildings[building].elec
print("finish CO");


###############################################################################################
# FTE Metric ##################################################################################
###############################################################################################


def fraction_energy_assigned_correctly_nan(predictions, ground_truth):
    predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)
    fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth.index = fraction_per_meter_ground_truth.index.map(lambda meter: meter.instance)
    fraction_per_meter_predictions.index = fraction_per_meter_predictions.index.map(lambda meter: meter.instance)
    fraction = 0
    for meter_instance in predictions_submeters.instance():
        #if math.isnan(fraction_per_meter_ground_truth[meter_instance]) == False:
        fraction += min(fraction_per_meter_ground_truth[meter_instance],
                        fraction_per_meter_predictions[meter_instance])
    return fraction

###############################################################################################
# Te Metric ###################################################################################
###############################################################################################
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
	
###############################################################################################
# Jaccard (Number of Appliances Identified Correctly ##########################################
###############################################################################################
def jaccard_similarity(predictions, ground_truth):
    predictions = predictions.drop(predictions.columns[[0]], axis=1); # remove column 1
    ground_truth = ground_truth.drop(ground_truth.columns[[0,1,2]], axis=1); # remove column 1, 2, 3
    predictions.columns = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'];
    ground_truth.columns = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'];
    pred_a_gt = predictions.astype(bool) & ground_truth.astype(bool);
    pred_u_gt = predictions.astype(bool) | ground_truth.astype(bool);
    sum_pred_a_gt = pred_a_gt.sum(axis=1);
    sum_pred_u_gt = pred_u_gt.sum(axis=1);
    ratio = sum_pred_a_gt / sum_pred_u_gt;
    Ja = ratio.sum() / len(ratio);
    return Ja;

### FHMM METRIC CALCULATION ###

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

# FTE
#print("starting FTE");
#FTE_fhmm = fraction_energy_assigned_correctly_nan(disag_fhmm_elec, test_elec)
#FTE_fhmm


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

# Te
#print("starting TE");
#Te_fhmm = total_disag_err(disag_fhmm_elec, test_elec)
#Te_fhmm

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

### CO METRIC CALCULATION ###

# FTE
print("starting FTE co");
FTE_co = fraction_energy_assigned_correctly_nan(disag_co_elec, test_elec)
print(FTE_co);


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

# Te
print("starting TE co");
Te_co = total_disag_err(disag_co_elec, test_elec);
print(Te_co);

# creating dataframe from both disaggregated and ground truth metergroups
disag_co_elec_df = disag_co_elec.dataframe_of_meters();
disag_co_elec_df_nona = disag_co_elec_df.dropna();
gt_full_df = test_elec.dataframe_of_meters();
gt_full_df_nona = gt_full_df.dropna();
gt_df_nona = gt_full_df_nona.ix[disag_co_elec_df_nona.index];

# Ja
print("starting Ja co");
Ja_co = jaccard_similarity(disag_co_elec_df_nona, gt_df_nona);
print(Ja_co);

###############################################################################################
# write disaggregation output format 1 ########################################################
###############################################################################################
# target = open("output_format.csv", 'w')
# size = disag_co_elec[1].load().next().axes[0].size
# for i in range(size):
#     data = ''
#     data += str(disag_co_elec[instance].load().next().axes[0][i].value)
#     data += ' '
#     for instance in disag_co_elec.submeters().instance():
#         if disag_co_elec[instance].load().next().ix[i][0] != 0:
#             #print(disag_co_elec[instance].label())
#             data += disag_co_elec[instance].label()
#             data += ' '
#             data += str(disag_co_elec[instance].load().next().ix[i][0])
#             data += ', '        
#     data += '\n'
#     #print(data)
#     target.write(data)
# 
# target.close()
# 
# ###############################################################################################
# # write disaggregation output format 2 (colomn) ###############################################
# ###############################################################################################
# target = open("output_format2.csv", 'w')
# 
# data = 'timestamp'
# data += '\t'    
# for instance in disag_fhmm_elec.submeters().instance():
#     data += disag_fhmm_elec[instance].label()            
#     data += '\t'
#     #print(data)
# 
# data += '\r\n'
# target.write(data)
# 
# size = disag_fhmm_elec[1].load().next().axes[0].size
# for i in range(size):
#     data = ''
#     data += str(disag_fhmm_elec[instance].load().next().axes[0][i].value)
#     data += '\t'
#     for instance in disag_fhmm_elec.submeters().instance():
#         data += str(disag_fhmm_elec[instance].load().next().ix[i][0])
#         data += '\t'        
#     
#     data += '\r\n'
#     #print(data)
#     target.write(data)
# 
# target.close()
# 
# ###############################################################################################
# # # check appliances state ####################################################################
# ###############################################################################################
# 
# target = open("output2.csv", 'w')
# 
# data = ''    
# for instance in disag_fhmm_elec.submeters().instance():
#     data += disag_fhmm_elec[instance].label()            
#     data += '\t'
#     #print(data)
# 
# data += '\r\n'
# target.write(data)
#     
# n11 = 0.0
# n10 = 0.0
# n01 = 0.0
# n00 = 0.0
# 
# size = disag_fhmm_elec[1].load().next().axes[0].size
# for i in range(int(50)):
#     #data = ''
#     for instance in disag_fhmm_elec.submeters().instance():
#         if disag_fhmm_elec.submeters()[instance].load().next().ix[i][0] != 0:
#             if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
#                 #data += '1'
#                 n11 = n11+1
#             else:
#                 n10 = n10+1       
# 	else:
#             #data += '0'
#             if test_elec.submeters()[instance].load().next().ix[i][0] != 0:
#                 #data += '1'
#                 n01 = n01+1
#             else:
#                 n00 = n00+1     
#     
#         #data += '\t'
#     #data += '\r\n'
#     #target.write(data)
#     #break
# 
# target.close()
