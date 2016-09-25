# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:39:36 2016

@author: neo
"""
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
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from datetime import datetime as dt
import sys
import argparse
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact
from nilmtk import utils
import re

def get_states(CO):
    appliances_name = [];
    state0 = [];
    state1 = [];
    state2 = [];
    for i in range(0,len(CO.model)):
        appliances_name.append(str(re.search("type='(.*?)'",str(CO.model[i]['training_metadata'].appliances)).group(1)).lower());
        state0.append(CO.model[i]['states'][0]);
        if(len(CO.model[i]['states'])==3):
            state1.append(CO.model[i]['states'][1]);
            state2.append(CO.model[i]['states'][2]);
        elif(len(CO.model[i]['states'])==2):
            state1.append(CO.model[i]['states'][1]);
            state2.append(CO.model[i]['states'][1]);
        else:
            state1.append(CO.model[i]['states'][0]);
            state2.append(CO.model[i]['states'][0]);
    state = pd.DataFrame(index=appliances_name, columns=['state0','state1','state2']);
    state.ix[:,'state0'] = state0;
    state.ix[:,'state1'] = state1;
    state.ix[:,'state2'] = state2;
    state.to_csv('states.csv');
    return state;

def groundtruth_generator(dataset_gt_df, dataset_test_df, building, co, occ_gt, occ_pred):
    dataset_gt_df.columns = [str(x).lower() for x in dataset_gt_df.columns.values];
    dataset_test_df.columns = [str(x).lower() for x in dataset_gt_df.columns.values];
    states = get_states(co);
    result_gt, ml_input_gt,result_gt, ml_input_gt = room_groundtruth(states, building, occ_gt, occ_pred, dataset_gt_df, dataset_test_df);
    return result_gt, ml_input_gt,result_gt, ml_input_gt;
    
def room_groundtruth(state, building, occ_gt, occ_pred, dataset_gt_df, dataset_test_df):
    occ_gt.columns=['occupancy'];
    occ_pred.columns=['occupancy'];
    dataset_gt_df.index.tz=None;
    yout_gt = dataset_gt_df.join(occ_gt,how='inner');
    dataset_test_df.index.tz=None;
    yout_gt.to_csv('yout_gt.csv');
    yout_test = dataset_test_df.join(occ_pred,how='inner');
    yout_test.to_csv('yout_test.csv');
    result_gt = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'],index=yout_gt.index);
    result_test = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'],index=yout_test.index);
    group_gt = pd.DataFrame(columns=map(str.upper, dataset_gt_df.columns),index=yout_gt.index);
    group_test = pd.DataFrame(columns=map(str.upper, dataset_test_df.columns),index=yout_test.index);
    single_gt = pd.DataFrame(columns=['mix'], index=yout_gt.index);
    single_test = pd.DataFrame(columns=['mix'], index=yout_test.index);
    group_gt.ix[:,:] = 0;
    result_gt.ix[:,:] = 0;
    single_gt.ix[:,:] = 0;
    group_test.ix[:,:] = 0;
    result_test.ix[:,:] = 0;
    single_test.ix[:,:] = 0;
    if building == 2:
        for i in yout_gt.index:
            #j = i.replace(tzinfo=CET)
            if (yout_gt.ix[i,'occupancy'] == 1): #
                if (yout_gt.ix[i,'kettle'] > 0) or (yout_gt.ix[i,'stove'] > 0) or (yout_gt.ix[i,'freezer'] >= int(state.ix['freezer','state2'])) or (yout_gt.ix[i,'fridge'] >= int(state.ix['fridge','state2'])) or (yout_gt.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                    result_gt.ix[i,'kitchen'] = 1;
                if (yout_gt.ix[i,'television'] >= int(state.ix['television','state2'])) or (yout_gt.ix[i,'audio system'] >= int(state.ix['audio system','state2'])) or (yout_gt.ix[i,'htpc'] >= int(state.ix['htpc','state2'])) or (yout_gt.ix[i,'lamp'] > int(state.ix['lamp','state2'])):
                    result_gt.ix[i,'livingroom'] = 1;
                if (yout_gt.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])) or (yout_gt.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])) or (yout_gt.ix[i,'tablet computer charger'] >= int(state.ix['tablet computer charger','state2'])):
                    result_gt.ix[i,'bedroom'] = 1;           
                result_gt.ix[i,'people'] = result_gt.ix[i,'kitchen'] + result_gt.ix[i,'livingroom'] + result_gt.ix[i,'bedroom'] + result_gt.ix[i,'bathroom'];
                if (result_gt.ix[i,'people'] >= 2):
                    result_gt.ix[i,'people'] = 2;
                    single_gt.ix[i,'mix'] = 1;
                else:
                    result_gt.ix[i,'people'] = 1;
                    single_gt.ix[i,'mix']=0;
            #if (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])):
                #group.ix[i,'fridge']=1;
                #sub_group.append('htpc');
                #group.ix[i,'freezer']=1;
                #sub_group.append('audio system');
            if (yout_gt.ix[i,'audio system'] >= int(state.ix['audio system','state2'])):
                #sub_group.append('fridge');
                group_gt.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                #sub_group.append('television');
            #if (yout.ix[i,'kettle'] > 0):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            if (yout_gt.ix[i,'television'] >= int(state.ix['television','state2'])):
                group_gt.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                group_gt.ix[i,'AUDIO SYSTEM']=1;
            #if (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            #if (y_on_off.ix[i,'freezer'] > 0):
                #sub_group.append('fridge');
            #if (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])):
                #group.ix[i,'freezer']=1;
            if (yout_gt.ix[i,'htpc'] >= int(state.ix['htpc','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group_gt.ix[i,'AUDIO SYSTEM']=1;
            if (yout_gt.ix[i,'lamp'] >= int(state.ix['lamp','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group_gt.ix[i,'HTPC']=1;
                group_gt.ix[i,'AUDIO SYSTEM']=1;
                group_gt.ix[i,'TELEVISION']=1;
            #if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                #sub_group.append('htpc');
                #sub_group.append('fridge');
                #sub_group.append('audio system');
            #if (yout.ix[i,'stove'] >= int(state.ix['stove','state2'])):
                #group.ix[i,'freezer']=1;
            #if (yout.ix[i,'tablet computer charger'] > int(state.ix['tablet computer charger','state2'])):
                #sub_group.append(fridge');
                #group.ix[i,'freezer']=1;
    result_gt.to_csv('groundtruth_room_.csv');
    for i in yout_test.index:
            #j = i.replace(tzinfo=CET)
            if (yout_test.ix[i,'occupancy'] == 1): #
                if (yout_test.ix[i,'kettle'] > 0) or (yout_test.ix[i,'stove'] > 0) or (yout_test.ix[i,'freezer'] >= int(state.ix['freezer','state2'])) or (yout_test.ix[i,'fridge'] >= int(state.ix['fridge','state2'])) or (yout_test.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                    result_test.ix[i,'kitchen'] = 1;
                if (yout_test.ix[i,'television'] >= int(state.ix['television','state2'])) or (yout_test.ix[i,'audio system'] >= int(state.ix['audio system','state2'])) or (yout_test.ix[i,'htpc'] >= int(state.ix['htpc','state2'])) or (yout_test.ix[i,'lamp'] > int(state.ix['lamp','state2'])):
                    result_test.ix[i,'livingroom'] = 1;
                if (yout_test.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])) or (yout_test.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])) or (yout_test.ix[i,'tablet computer charger'] >= int(state.ix['tablet computer charger','state2'])):
                    result_test.ix[i,'bedroom'] = 1;           
                result_test.ix[i,'people'] = result_test.ix[i,'kitchen'] + result_test.ix[i,'livingroom'] + result_test.ix[i,'bedroom'] + result_test.ix[i,'bathroom'];
                if (result_test.ix[i,'people'] >= 2):
                    result_test.ix[i,'people'] = 2;
                    single_test.ix[i,'mix'] = 1;
                else:
                    result_test.ix[i,'people'] = 1;
                    single_test.ix[i,'mix']=0;
            #if (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])):
                #group.ix[i,'fridge']=1;
                #sub_group.append('htpc');
                #group.ix[i,'freezer']=1;
                #sub_group.append('audio system');
            if (yout_test.ix[i,'audio system'] >= int(state.ix['audio system','state2'])):
                #sub_group.append('fridge');
                group_test.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                #sub_group.append('television');
            #if (yout.ix[i,'kettle'] > 0):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            if (yout_test.ix[i,'television'] >= int(state.ix['television','state2'])):
                group_test.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                group_test.ix[i,'AUDIO SYSTEM']=1;
            #if (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            #if (y_on_off.ix[i,'freezer'] > 0):
                #sub_group.append('fridge');
            #if (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])):
                #group.ix[i,'freezer']=1;
            if (yout_test.ix[i,'htpc'] >= int(state.ix['htpc','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group_test.ix[i,'AUDIO SYSTEM']=1;
            if (yout_test.ix[i,'lamp'] >= int(state.ix['lamp','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group_test.ix[i,'HTPC']=1;
                group_test.ix[i,'AUDIO SYSTEM']=1;
                group_test.ix[i,'TELEVISION']=1;
            #if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                #sub_group.append('htpc');
                #sub_group.append('fridge');
                #sub_group.append('audio system');
            #if (yout.ix[i,'stove'] >= int(state.ix['stove','state2'])):
                #group.ix[i,'freezer']=1;
            #if (yout.ix[i,'tablet computer charger'] > int(state.ix['tablet computer charger','state2'])):
                #sub_group.append(fridge');
                #group.ix[i,'freezer']=1;
    result_test.to_csv('test_room_.csv');
    ml_input_gt = yout_gt.join(group_gt, how='inner');
    ml_input_gt = ml_input_gt.join(single_gt, how='inner');
    ml_input_gt.to_csv('ml_input_gt.csv');
    ml_input_test = yout_test.join(group_test, how='inner');
    ml_input_test = ml_input_test.join(single_test, how='inner');
    ml_input_test.to_csv('ml_input_test.csv');
    return result_gt, ml_input_gt,result_gt, ml_input_gt;

def extract_occ(file1,file2, frequency):
    occ_raw  = pd.read_csv(filepath_or_buffer=file1,skiprows=0,sep=',');
    occ_raw2 = pd.read_csv(filepath_or_buffer=file2,skiprows=0,sep=',');
    occ_data = pd.DataFrame(data=None, columns=occ_raw.columns);
    occ_data = occ_data.append(occ_raw2);
    indexed_occupancy = occ_data.set_index(['Unnamed: 0']);
    occupancy_col = [];
    occupancy_index = [];
    for i in indexed_occupancy.index.values:
        for j in indexed_occupancy.columns.values:
            #print(i+ ' '+j);
            temp = pd.to_datetime(str(i)+ " " + str(j), format="%d-%b-%Y '%H:%M:%S'");
            occupancy_index.append(temp);
            occupancy_col.append(indexed_occupancy.ix[i,j]);
    result = pd.DataFrame(index=occupancy_index,data=occupancy_col, columns=['occupancy']);
    result = result.resample(str(frequency) + 'S').mean().round(0);
    result.to_csv('resample_occupancy.csv');
    return result;

def feature_generator(dataset_loc, start_time, end_time, building,freq, occ_data):
    label = [];
    label_upper= [];
    data = DataSet(dataset_loc);
    data.set_window(start=start_time, end=end_time);
    data_elec = data.buildings[building].elec;
    for i in data_elec.submeters().instance():
        label.append(str(data_elec[i].label()).lower());
        label_upper.append(str(data_elec[i].label()).upper());
    test_elec_df = data_elec.dataframe_of_meters().resample(str(freq)+'S').max().round(0);
    test_elec_df = test_elec_df.drop(test_elec_df.columns[[0]], axis=1);
    test_elec_df.columns = label;
    test_elec_df.to_csv('feature_elec.csv');
    states = pd.DataFrame.from_csv('states.csv');
    result = room_feature(states, building, label_upper, occ_data);
    return result;
    
def room_feature(state, housing, label_upper, occ_data):
    occupancy_gt = occ_data;
    yout = pd.DataFrame.from_csv('feature_elec.csv');
    single = [];
    yout = yout.join(occupancy_gt,how='inner');
    yout.to_csv('nilmtk_occupancy_join_feature.csv');
    result = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'],index=yout.index);
    group = pd.DataFrame(columns=label_upper,index=yout.index);
    single = pd.DataFrame(columns=['mix'], index=yout.index);
    group.ix[:,:] = 0;
    result.ix[:,:] = 0;
    single.ix[:,:] = 0;
    if housing == 2:
        for i in yout.index:
            #j = i.replace(tzinfo=CET)
            if (yout.ix[i,'occupancy'] == 1): #
                if (yout.ix[i,'kettle'] > 0) or (yout.ix[i,'stove'] > 0) or (yout.ix[i,'freezer'] >= int(state.ix['freezer','state2'])) or (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])) or (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                    result.ix[i,'kitchen'] = 1;
                if (yout.ix[i,'television'] >= int(state.ix['television','state2'])) or (yout.ix[i,'audio system'] >= int(state.ix['audio system','state2'])) or (yout.ix[i,'htpc'] >= int(state.ix['htpc','state2'])) or (yout.ix[i,'lamp'] > int(state.ix['lamp','state2'])):
                    result.ix[i,'livingroom'] = 1;
                if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])) or (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])) or (yout.ix[i,'tablet computer charger'] >= int(state.ix['tablet computer charger','state2'])):
                    result.ix[i,'bedroom'] = 1;           
                result.ix[i,'people'] = result.ix[i,'kitchen'] + result.ix[i,'livingroom'] + result.ix[i,'bedroom'] + result.ix[i,'bathroom'];
                if (result.ix[i,'people'] >= 2):
                    result.ix[i,'people'] = 2;
                    single.ix[i,'mix'] = 1;
                else:
                    result.ix[i,'people'] = 1;
                    single.ix[i,'mix']=0;
            #if (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])):
                #group.ix[i,'fridge']=1;
                #sub_group.append('htpc');
                #group.ix[i,'freezer']=1;
                #sub_group.append('audio system');
            if (yout.ix[i,'audio system'] >= int(state.ix['audio system','state2'])):
                #sub_group.append('fridge');
                group.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                #sub_group.append('television');
            #if (yout.ix[i,'kettle'] > 0):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            if (yout.ix[i,'television'] >= int(state.ix['television','state2'])):
                group.ix[i,'HTPC']=1;
                #group.ix[i,'freezer']=1;
                group.ix[i,'AUDIO SYSTEM']=1;
            #if (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
            #if (y_on_off.ix[i,'freezer'] > 0):
                #sub_group.append('fridge');
            #if (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])):
                #group.ix[i,'freezer']=1;
            if (yout.ix[i,'htpc'] >= int(state.ix['htpc','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group.ix[i,'AUDIO SYSTEM']=1;
            if (yout.ix[i,'lamp'] >= int(state.ix['lamp','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                group.ix[i,'HTPC']=1;
                group.ix[i,'AUDIO SYSTEM']=1;
                group.ix[i,'TELEVISION']=1;
            #if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])):
                #sub_group.append('fridge');
                #group.ix[i,'freezer']=1;
                #sub_group.append('htpc');
                #sub_group.append('fridge');
                #sub_group.append('audio system');
            #if (yout.ix[i,'stove'] >= int(state.ix['stove','state2'])):
                #group.ix[i,'freezer']=1;
            #if (yout.ix[i,'tablet computer charger'] > int(state.ix['tablet computer charger','state2'])):
                #sub_group.append(fridge');
                #group.ix[i,'freezer']=1;
    result.to_csv('groundtruth_room_.csv');
    ml_input = yout.join(group, how='inner');
    ml_input = ml_input.join(single, how='inner')
    ml_input.to_csv('ml_input.csv');
    return result, ml_input;

#loc = '/home/neo/NILMTK_experimental/eco1.h5';
#start = "07-01-2012";
#end = "09-30-2012";
#result, ml_input = groundtruth_generator(loc,start,end,2,900);