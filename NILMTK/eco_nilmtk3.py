from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
#%matplotlib inline
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation
start = time.time()
from nilmtk.disaggregate import fhmm_exact
fhmm = fhmm_exact.FHMM()


train = DataSet('data/eco.h5')
test = DataSet('data/eco.h5')

train_elec = train.buildings[1].elec

top_5_train_elec = train_elec.submeters().select_top_k(k=5)
# Note that we have given the sample period to downsample the data to 1 minute
fhmm.train(top_5_train_elec, sample_period=60)
end = time.time()
print("Runtime =", end-start, "seconds.")
