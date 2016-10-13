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

from metadata.metadata import Metadata

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


###############################################################################################
# Jaccard (Number of Appliances Identified Correctly ##########################################
###############################################################################################

def jaccard_similarity(predictions, ground_truth):
    predictions = predictions.drop(predictions.columns[[0]], axis=1) 
    ground_truth = ground_truth.drop(ground_truth.columns[[0,1,2]], axis=1)
    predictions.columns = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    ground_truth.columns = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    pred_a_gt = predictions.astype(bool) & ground_truth.astype(bool)
    pred_u_gt = predictions.astype(bool) | ground_truth.astype(bool)
    sum_pred_a_gt = pred_a_gt.sum(axis=1)
    sum_pred_u_gt = pred_u_gt.sum(axis=1)
    ratio = sum_pred_a_gt / sum_pred_u_gt
    Ja = ratio.sum() / len(ratio)
    return Ja

def jaccard_similarity_b2_k8(predictions, ground_truth):
    predictions = predictions.drop(predictions.columns[[0]], axis=1) 
    ground_truth = ground_truth.drop(ground_truth.columns[[0,1,2,3,5,10,11]], axis=1)
    predictions.columns = ['5', '7', '8', '9', '10', '13', '14', '15']
    ground_truth.columns = ['5', '7', '8', '9', '10', '13', '14', '15']
    pred_a_gt = predictions.astype(bool) & ground_truth.astype(bool)
    pred_u_gt = predictions.astype(bool) | ground_truth.astype(bool)
    sum_pred_a_gt = pred_a_gt.sum(axis=1)
    sum_pred_u_gt = pred_u_gt.sum(axis=1)
    ratio = sum_pred_a_gt / sum_pred_u_gt
    Ja = ratio.sum() / len(ratio)
    return Ja

### FHMM METRIC CALCULATION ###
period_s = 60

#load train and test data
total = DataSet('/media/airawan/DATA/Data/eco.h5')
train = DataSet('/media/airawan/DATA/Data/eco.h5')
test = DataSet('/media/airawan/DATA/Data/eco.h5')

# train and test visualization
#train.buildings[1].elec.mains().plot()
#test.buildings[1].elec.mains().plot()
#plt.show()

# change this variable to get the result from different building
building = 2

# start 2012-06-01 end 2013-02-01
tf_total = total.buildings[building].elec.mains().get_timeframe()

# train and test interval
if building == 1:
    train.set_window(start="08-01-2012", end="09-30-2012")
    #test.set_window(start="30-09-2012", end="30-11-2012")
    #train.set_window(start="08-01-2012", end="09-30-2012")
    test.set_window(start="06-01-2012", end="06-07-2012")
    
    print(1)
elif building == 2:
    #train.set_window(start="06-01-2012", end="06-30-2012")
    #test.set_window(start="06-30-2012", end="07-31-2012")
    #train.set_window(start="06-07-2012", end="06-30-2012")
    #test.set_window(start="06-01-2012", end="06-07-2012")
    
    #train windows 3 train 3/4 month test 1 month
    train.set_window(start="06-02-2012", end="06-20-2012")
    test.set_window(start="06-21-2012", end="07-20-2012")
    
    #train windows 2 day
    #train.set_window(start="06-02-2012", end="06-20-2012")
    #test.set_window(start="06-20-2012", end="06-21-2012")
    #train.set_window(start="06-20-2012", end="06-22-2012")
    
    print(2)
elif building == 3:
    #train.set_window(start="11-1-2012", end="11-30-2012")
    #test.set_window(start="11-30-2012", end="12-31-2012")
    test.set_window(start="1-1-2013", end="1-7-2013")
    train.set_window(start="12-07-2012", end="12-31-2012")
    print(3)

total_elec = total.buildings[building].elec
train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

tf_train = train.buildings[building].elec.mains().get_timeframe()
tf_test = test.buildings[building].elec.mains().get_timeframe()

tf_train
tf_test

# CO disaggregation test
start = time.time()
from nilmtk.disaggregate import CombinatorialOptimisation
co = CombinatorialOptimisation()
# Note that we have given the sample period to downsample the data to 15 minute
#co.train(top_8_train_elec, sample_period=period_s)
#co.train(train_elec.submeters(), sample_period=period_s)
meta_eco = Metadata("ECO").centroids
co.train_centroid(train_elec.submeters(), centroids = meta_eco)
end = time.time()
print("Runtime =", end-start, "seconds.")

for i, model in enumerate(co.model):
    print(model['training_metadata'].instance(), train_elec[model['training_metadata'].instance()].label(), model['states'])

start = time.time()
disag_filename = '/media/airawan/DATA/Data/eco-b2-kall-co-2d-prio-1m.h5'
#disag_filename = '/media/airawan/DATA/Data/eco-b2-kall-co-0.75:1-15m.h5'
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
print(FTE_co);
# Te
Te_co = total_disag_err(disag_co_elec, test_elec)
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

for i in range(12):
    plt.clf()
    test_elec[i+4].plot()
    disag_co_elec[i+4].plot()
    fig_name = test_elec[i+4].label()
    plt.savefig('/media/airawan/DATA/Data/plot' + fig_name + ' 1d.png')

