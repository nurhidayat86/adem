# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 22:44:33 2016
Convert data ECO
@author: neo
"""
from __future__ import print_function, division
import time
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.metrics import f1_score
#from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.disaggregate import fhmm_exact
from matplotlib import rcParams
import matplotlib.pyplot as plt
#%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

ecopath = '/home/neo/ECO/'
train = DataSet('/home/neo/ECO/eco.h5')
test = DataSet('/home/neo/ECO/eco.h5')
building = 1
fhmm = fhmm_exact.FHMM()

#Splitting train and test data
train.set_window(start="01-08-2012", end="31-10-2012")
test.set_window(start="01-11-2012")

train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

print("\n\ntrain data")
print(train_elec)
print("\n\ntest data")
print(test_elec)

#training with FHMM algortihm, with sampe period = 60 seconds 
sub_train_elec = train_elec.submeters()
start = time.time()
fhmm.train(sub_train_elec, sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")

#saving diassagregation dataset
disag_filename = '/home/neo/ECO/fhmm_eco.h5'
output = HDFDataStore(disag_filename, 'w')
fhmm.disaggregate(test_elec.mains(), output, sample_period=60)
output.close()

#Open disaggregation dataset and plot it
disag_fhmm = DataSet(disag_filename)
disag_fhmm_elec = disag_fhmm.buildings[building].elec
f1_fhmm = f1_score(disag_fhmm_elec, test_elec)
f1_fhmm.index = disag_fhmm_elec.get_labels(f1_fhmm.index)
f1_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("FHMM");

#Plotting train data
#train_elec.mains().plot()
#test_elec.mains().plot()
