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

def groupmix_rlo(state, label_upper, train_elec_df):
    yout = train_elec_df;
    yout.index.tz = None;
    yout = yout.shift(periods=2, freq='H');
    single = [];
    result = pd.DataFrame(columns=['kitchen','livingroom','bedroom','bathroom','people'], index=yout.index);
    group = pd.DataFrame(columns=label_upper,index=yout.index);
    single = pd.DataFrame(columns=['mix'], index=yout.index);
    group.ix[:,:] = 0;
    result.ix[:,:] = 0;
    single.ix[:,:] = 0;
	
    for i in yout.index:

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
        if (yout.ix[i,'audio system'] >= int(state.ix['audio system','state2'])):
            group.ix[i,'HTPC']=1;
        if (yout.ix[i,'television'] >= int(state.ix['television','state2'])):
            group.ix[i,'HTPC']=1;
            group.ix[i,'AUDIO SYSTEM']=1;
        if (yout.ix[i,'htpc'] >= int(state.ix['htpc','state2'])):
            group.ix[i,'AUDIO SYSTEM']=1;
        if (yout.ix[i,'lamp'] >= int(state.ix['lamp','state2'])):
            group.ix[i,'HTPC']=1;
            group.ix[i,'AUDIO SYSTEM']=1;
            group.ix[i,'TELEVISION']=1;
    ml_input = yout.join(group, how='inner');
    ml_input = ml_input.join(single, how='inner');    
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
    return state;
	
def groupmix_rlo_generator(dataset_loc, start_time, end_time, freq, co):
    building = 2;
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
    states = get_states(co);
    group_mix, room_occ_num_people = groupmix_rlo(states, label_upper, train_elec_df);
    return group_mix, room_occ_num_people;
