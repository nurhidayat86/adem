from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

from nilmtk.electric import align_two_meters
from nilmtk.metergroup import iterate_through_submeters_of_two_metergroups    
import numpy as np
import pandas as pd

# based on method : def mean_normalized_error_power(predictions, ground_truth):
def total_disag_err(predictions, ground_truth):
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)   
    # additional of total variable
    total_appliances_power = 0.0  
    total_ground_truth_power = 0.0 
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        total_abs_diff = 0.0
        sum_of_ground_truth_power = 0.0
        for aligned_meters_chunk in align_two_meters(pred_meter, ground_truth_meter):
            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
            total_abs_diff += sum(abs(diff.dropna()))
	    total_ground_truth_power += sum_of_ground_truth_power
	    total_appliances_power += total_abs_diff
    return total_appliances_power/total_ground_truth_power

# load train and test data
train = DataSet('/media/airawan/DATA/Data/eco.h5')
test = DataSet('/media/airawan/DATA/Data/eco.h5')

# train and test visualization
#train.buildings[1].elec.mains().plot()
#test.buildings[1].elec.mains().plot()
#plt.show()

# change this variable to get the result from different building
building = 1

# start 2012-06-01 end 2013-02-01
tf_total = train.buildings[building].elec.mains().get_timeframe()

#1 months training, 7 months test
train.set_window(end="31-07-2012")
test.set_window(start="31-07-2012")

train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

# mains meters frame checking
#train_elec.mains().plot()
#plt.show()
#test_elec.mains().plot()
#plt.show()

# check data sampling time
#fridge_meter = train_elec['fridge', 1]
#fridge_df = fridge_meter.load().next()
#fridge_df.head()

# select top 5 submeter
top_5_train_elec = train_elec.submeters().select_top_k(k=5)
top_5_train_elec

# FHMM
start = time.time()
from nilmtk.disaggregate import fhmm_exact
fhmm = fhmm_exact.FHMM()
# Note that we have given the sample period to downsample the data to 1 minute
fhmm.train(top_5_train_elec, sample_period=60)
#fhmm.train(train_elec.submeters(), sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")

disag_filename = '/media/airawan/DATA/Data/eco1-fhmm.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
fhmm.disaggregate(test_elec.mains(), output, sample_period=60)
output.close()

disag_fhmm = DataSet(disag_filename)
disag_fhmm_elec = disag_fhmm.buildings[building].elec

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

# metric FTE
from nilmtk.metrics import fraction_energy_assigned_correctly
FTE_fhmm = fraction_energy_assigned_correctly(disag_fhmm_elec, test_elec)
FTE_fhmm

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
Te_fhmm = total_disag_err(disag_fhmm_elec, test_elec)
Te_fhmm

# CO disaggregation test
start = time.time()
from nilmtk.disaggregate import CombinatorialOptimisation
co = CombinatorialOptimisation()
# Note that we have given the sample period to downsample the data to 1 minute
co.train(top_5_train_elec, sample_period=60)
#co.train(train_elec, sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")

disag_filename = '/media/airawan/DATA/Data/eco1-co.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
co.disaggregate(test_elec.mains(), output, sample_period=60)
output.close()

# print(train_elec.mains().load().next().head())

disag_co = DataSet(disag_filename)
disag_co_elec = disag_co.buildings[building].elec

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

# FTE
FTE_co = fraction_energy_assigned_correctly(disag_co_elec, test_elec)
FTE_co


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
Te_co = total_disag_err(disag_co_elec, test_elec)
Te_co

# results print out 
FTE_fhmm
FTE_co

Te_fhmm
Te_co

# write disaggregation output, took quite long time
target = open("output_format.txt", 'w')
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


