from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

train = DataSet('/media/airawan/DATA/Data/eco.h5')
test = DataSet('/media/airawan/DATA/Data/eco.h5')

#train.buildings[1].elec.mains().plot()
#plt.show()

building = 1

# start 2012-06-01 end 2013-02-01
tf_total = train.buildings[building].elec.mains().get_timeframe()

#2 months training, 6 months test
train.set_window(end="31-07-2012")
test.set_window(start="31-07-2012")

train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

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
end = time.time()
print("Runtime =", end-start, "seconds.")

disag_filename = '/media/airawan/DATA/Data/eco1-fhmm.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 60 seconds
fhmm.disaggregate(test_elec.mains(), output, sample_period=60)
output.close()

disag_fhmm = DataSet(disag_filename)
disag_fhmm_elec = disag_fhmm.buildings[building].elec

"""
from nilmtk.metrics import f1_score
f1_fhmm = f1_score(disag_fhmm_elec, test_elec)
f1_fhmm.index = disag_fhmm_elec.get_labels(f1_fhmm.index)
f1_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("FHMM");
"""

from nilmtk.metrics import fraction_energy_assigned_correctly
FTE_fhmm = fraction_energy_assigned_correctly(disag_fhmm_elec, test_elec)
FTE_fhmm

from nilmtk.metrics import mean_normalized_error_power
Te_fhmm = mean_normalized_error_power(disag_fhmm_elec, test_elec)
Te_fhmm.index = disag_fhmm_elec.get_labels(Te_fhmm.index)
Te_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('Te');
plt.title("FHMM");
plt.show()

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
from nilmtk.metrics import f1_score
f1_co= f1_score(disag_co_elec, test_elec)
f1_co.index = disag_co_elec.get_labels(f1_co.index)
f1_co.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("CO");
"""

FTE_co = fraction_energy_assigned_correctly(disag_co_elec, test_elec)
FTE_co

Te_co = mean_normalized_error_power(disag_co_elec, test_elec)
Te_co.index = disag_co_elec.get_labels(Te_co.index)
Te_co.plot(kind='barh')
Te_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('Te');
plt.title("CO and FHMM");
plt.show()


