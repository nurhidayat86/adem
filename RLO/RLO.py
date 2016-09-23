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

def room_groundtruth(state, yout, housing, label_upper):
    result = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'],index=yout.index);
    label = yout.columns.values;
    group = pd.DataFrame(columns=label_upper,index=yout.index)   
    single = [];
    group.ix[:,:] = 0;
    if housing == 2:
        for i in yout.index:
            result.ix[i,:] = 0;
            if (yout.ix[i,'kettle'] > 0) or (yout.ix[i,'stove'] > 0) or (yout.ix[i,'freezer'] >= int(state.ix['freezer','state2'])) or (yout.ix[i,'fridge'] >= int(state.ix['fridge','state2'])) or (yout.ix[i,'dish washer'] >= int(state.ix['dish washer','state2'])):
                result.ix[i,'kitchen'] = 1;
            if (yout.ix[i,'television'] >= int(state.ix['television','state2'])) or (yout.ix[i,'audio system'] >= int(state.ix['audio system','state2'])) or (yout.ix[i,'htpc'] >= int(state.ix['htpc','state2'])) or (yout.ix[i,'lamp'] > int(state.ix['lamp','state2'])):
                result.ix[i,'livingroom'] = 1;
            if (yout.ix[i,'laptop computer'] >= int(state.ix['laptop computer','state2'])) or (yout.ix[i,'air handling unit'] >= int(state.ix['air handling unit','state2'])) or (yout.ix[i,'tablet computer charger'] >= int(state.ix['tablet computer charger','state2'])):
                result.ix[i,'bedroom'] = 1;           
            result.ix[i,'people'] = result.ix[i,'kitchen'] + result.ix[i,'livingroom'] + result.ix[i,'bedroom'] + result.ix[i,'bathroom'];
            if (result.ix[i,'people'] >= 2):
                result.ix[i,'people'] = 2;
                single.append(False);
            else:
#               print('Somewhere');
                result.ix[i,'people'] = 1;
                single.append(True);
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
    group['mix'] = single;
    ml_input = yout.join(group, how='inner');
    ml_input.to_csv('ml_input.csv');
    return result, ml_input;

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

def groundtruth_generator(dataset_loc, start_time, end_time, building,freq):
    label = [];
    label_upper= [];
    data = DataSet(dataset_loc);
    data.set_window(start=start_time, end=end_time);
    data_elec = data.buildings[building].elec;
    for i in data_elec.submeters().instance():
        label.append(str(data_elec[i].label()).lower());
        label_upper.append(str(data_elec[i].label()).upper());
    train_elec_df = data_elec.dataframe_of_meters().resample(str(freq)+'S').max().round(0);
    train_elec_df = train_elec_df.drop(train_elec_df.columns[[0,1,2]], axis=1);
    train_elec_df.columns = label;
    train_elec_df.to_csv('groundtruth_elec.csv')
    co = CombinatorialOptimisation();
    co.train(data_elec.submeters(), sample_period=freq);
    states = get_states(co);
    result = room_groundtruth(states, train_elec_df, building, label_upper);
    return result;


loc = '/home/neo/NILMTK_experimental/eco1.h5';
start = "07-01-2012";
end = "09-30-2012"
result, ml_input = groundtruth_generator(loc,start,end,2,900);