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


# based on method : def mean_normalized_error_power(predictions, ground_truth):
def total_disag_err(predictions, ground_truth):
    #only iterate for the instance in the prediction/ elecmeter with lesser instance
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
	    sum_of_ground_truth_power += aligned_meters_chunk.icol(1).sum()
            total_ground_truth_power += sum_of_ground_truth_power
	    total_appliances_power += total_abs_diff
    return float(total_appliances_power)/float(total_ground_truth_power)


#load train and test data
total = DataSet('/media/airawan/DATA/Data/eco.h5')
train = DataSet('/media/airawan/DATA/Data/eco.h5')
test = DataSet('/media/airawan/DATA/Data/eco.h5')

# train and test visualization
#train.buildings[1].elec.mains().plot()
#test.buildings[1].elec.mains().plot()
#plt.show()

# change this variable to get the result from different building
building = 1

# start 2012-06-01 end 2013-02-01
tf_total = total.buildings[building].elec.mains().get_timeframe()

# train and test interval
if building == 1:
    train.set_window(start="08-01-2012", end="02-09-2012")
    test.set_window(start="30-09-2012")
    print(1)
elif building == 2:
    train.set_window(start="01-07-2012", end="30-09-2012")
    test.set_window(start="30-09-2012")
    print(2)
elif building == 3:
    train.set_window(start="1-11-2012", end="30-11-2012")
    test.set_window(start="30-11-2012", end="1-12-2012")
    print(3)

total_elec = total.buildings[building].elec
train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

tf_train = train.buildings[building].elec.mains().get_timeframe()
tf_test = test.buildings[building].elec.mains().get_timeframe()

tf_train
tf_test

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
top_5_train_elec = train_elec.submeters().select_top_k(k=9)
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

start = time.time()
disag_filename = '/media/airawan/DATA/Data/eco-b2-k9-fhmm-30sep.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
fhmm.disaggregate(test_elec.mains(), output, sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")
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
FTE_fhmm = fraction_energy_assigned_correctly_nan(disag_fhmm_elec, test_elec)
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
#co.train(top_5_train_elec, sample_period=60)
co.train(train_elec.submeters(), sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")

start = time.time()
disag_filename = '/media/airawan/DATA/Data/eco-b1-k7-co-30sep.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
co.disaggregate(test_elec.mains(), output, sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")
output.close()

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
FTE_co = fraction_energy_assigned_correctly_nan(disag_co_elec, test_elec)
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
target = open("output_format2.csv", 'w')

data = 'timestamp'
data += '\t'    
for instance in disag_fhmm_elec.submeters().instance():
    data += disag_fhmm_elec[instance].label()            
    data += '\t'
    #print(data)

data += '\r\n'
target.write(data)

size = disag_fhmm_elec[1].load().next().axes[0].size
for i in range(size):
    data = ''
    data += str(disag_fhmm_elec[instance].load().next().axes[0][i].value)
    data += '\t'
    for instance in disag_fhmm_elec.submeters().instance():
        data += str(disag_fhmm_elec[instance].load().next().ix[i][0])
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



