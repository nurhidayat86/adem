from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

from nilmtk.electric import align_two_meters
from nilmtk.metergroup import iterate_through_submeters_of_two_metergroups    
import numpy as np
import pandas as pd
import math

################################################################################################
############################# Metric: Fraction Energy Assigned Correctly #######################
################################################################################################

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

################################################################################################
############################# Metric: Total Disaggregation Error ###############################
################################################################################################

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

################################################################################################
############################# Metric: Jaccard Similarity #######################################
################################################################################################

def jaccard_similarity(predictions, ground_truth, submeters_instance, gt_instance):
    predictions = predictions.drop(predictions.columns[[0]], axis=1)
    ind = submeters_instance 
    ind = list(map(lambda x: x - 1, ind))
    ind_test = gt_instance 
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

################################################################################################
############################# NILMTK DRED disaggregation ########################################
################################################################################################

def nilmtkDREDfunc(dataset_loc, train_start, train_end, test_start, test_end, output_period):
    #### configuration ####
    period_s = output_period;
    building = 1;
    #### load ####
    total = DataSet(dataset_loc);
    train = DataSet(dataset_loc);
    test = DataSet(dataset_loc);
    train.set_window(start=train_start, end=train_end);
    test.set_window(start=test_start, end=test_end);
    print(train_start);
    print(train_end);
    print(test_start);
    print(test_end);
    #### get timeframe ####
    tf_total = total.buildings[building].elec.mains().get_timeframe();
    tf_train = train.buildings[building].elec.mains().get_timeframe();
    tf_test = test.buildings[building].elec.mains().get_timeframe();
    #### eletrical metergroup ####
    total_elec = total.buildings[building].elec;
    train_elec = train.buildings[building].elec;
    test_elec = test.buildings[building].elec;
    #### training process ####
    start = time.time();
    from nilmtk.disaggregate import CombinatorialOptimisation;
    co = CombinatorialOptimisation();
    co.train(train_elec, sample_period=period_s);
    end = time.time();
    print("Runtime =", end-start, "seconds.");
    #### disaggregation process ####
    start = time.time();
    disag_filename = dataset_loc + 'DREDapp.h5';
    output = HDFDataStore(disag_filename, 'w');
    co.disaggregate(test_elec.mains(), output, sample_period=period_s);
    end = time.time();
    print("Runtime =", end-start, "seconds.");
    output.close();
    disag_co = DataSet(disag_filename);
    disag_co_elec = disag_co.buildings[building].elec;
    #### creating dataframe from both disaggregated and ground truth metergroups
    disag_co_elec_df = disag_co_elec.dataframe_of_meters();
    gt_full_df = test_elec.dataframe_of_meters();
    # drop the NA, it might be needed (initially it is used for Ja)
    disag_co_elec_df_nona = disag_co_elec_df.dropna();
    gt_full_df_nona = gt_full_df.dropna();
    # drop the unwanted timestamp    
    gt_df_nona = gt_full_df_nona.ix[disag_co_elec_df_nona.index];
    #### output ####
    # drop aggregated power from output
    disag_co_elec_submeter_df = disag_co_elec_df.drop(disag_co_elec_df.columns[[0]], axis=1);
    # drop the unwanted timestamp on ground truth (take the sampled timestamp)
    gt_df_aligned = gt_full_df.ix[disag_co_elec_submeter_df.index];
    # drop aggregated power from ground truth
    gt_df_sub = gt_df_aligned.drop(gt_df_aligned.columns[[0]], axis=1);
    # train data frame, resample based in disaggregation period, drop the main power
    train_elec_df = train_elec.dataframe_of_meters();
    train_elec_df_aligned = train_elec_df.resample(str(period_s)+'S').asfreq()[0:];
    train_elec_df_aligned_drop = train_elec_df_aligned.drop(train_elec_df_aligned.columns[[0]], axis=1)
    return disag_co_elec_submeter_df, gt_df_sub, co, train_elec_df_aligned_drop;
