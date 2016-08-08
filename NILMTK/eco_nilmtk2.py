from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
#%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

train = DataSet('data/eco.h5')
test = DataSet('data/eco.h5')
building = 1

plt.show (train.buildings[building].elec.mains().plot())

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")
train_elec = train.buildings[building].elec
print (train_elec)
test_elec = test.buildings[building].elec
print (test_elec)

plt.show(train_elec.mains().plot())
plt.show (test_elec.mains().plot())
hair_meter = train_elec['hair dryer']
hair_df = hair_meter.load().next()
print (hair_df.head())

#Training and Disaggregation


