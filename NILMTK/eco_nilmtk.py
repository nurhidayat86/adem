"""
author : Rizky Dharmawan
ver0.1 : 1 Aug 2016;
ver0.2 : 4 Aug 2016;
ver0.3 : 5 Aug 2016;

Description : 
Using ECO (Electricity Consumption & Occupancy) data set, the output of ECO NILMTK dataset will be used
 for the input of Occupancy.The output results of NILMTK that will be used for occupancy is Energy 
 disaggregation for each applliances. 

NILMTK : https://github.com/nilmtk/nilmtk/tree/master/docs/manual
consists of 9 parts : 1) Introduction, 2) Installation, 3) Data (Convert and load data), 
4) The Load API, 5) Meter selection & basic statistic, 6) Out of cores,preprocessing & more stats, 
7) Disaggregation and Metrics, 8) Cookbook, 9) and Ipython notebook demos 

"""
###############################################################################################
# 3.Convert Data NILMTK #######################################################################
###############################################################################################
# Open HDF5 in NILMTK
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nilmtk.utils import print_dict
from nilmtk import DataSet
eco = DataSet('data/eco.h5')

"""
Choose building that will be shown (building number 1/2/3, etc) and then it will shows the
Metergroup : SmartMeter and Appliances
"""
print eco.buildings[1].elec #Each building has an elec attribute which is a MeterGroup object 

###############################################################################################
# 4. The Load API: Loading data into memory ###################################################
###############################################################################################
elec = eco.buildings[1].elec

"""
Select Appliances (fridge/hair dayer/coffe maker/kettle/washing machine/computer/freezer)
from the each building

However there are errors for some appliances such as : fridge, coffe maker, and freezer.
The error messages shown below :
Traceback (most recent call last):
  File "eco_nilmtk.py", line 42, in <module>
    fridge = elec['fridge']
  File "/home/rizky/nilmtk/nilmtk/metergroup.py", line 198, in __getitem__
    return self[(key, 1)]
  File "/home/rizky/nilmtk/nilmtk/metergroup.py", line 242, in __getitem__
    return self[{'type': key[0], 'instance': key[1]}]
  File "/home/rizky/nilmtk/nilmtk/metergroup.py", line 259, in __getitem__
    .format(len(meters)))
Exception: search terms match 2 appliances
""" 
#fridge = elec['fridge'] #error
hair = elec['hair dryer'] 
#coffe = elec['coffe maker'] #error
#kettle = elec['kettle'] 
#washing = elec['washing machine']
#computer = elec['computer'] 
#freezer = elec['freezer'] #error

#See the measurement from the appliances
#print fridge.available_columns() #error
print hair.available_columns() 
#print coffe.available_columns() #error 
#print kettle.available_columns() 
#print washing.available_columns()  
#print computer.available_columns()  
#print freezer.available_columns() #error

###############################################################################################
# 4.1 Loading data all columns ################################################################
###############################################################################################
"""
Description for Loading data all columns :
in this example, hair appliance is used
a. To show load all columns 
   hair.load().next()
b. Specify physical quantity or AC type 
   hair.load(physical_quantity='power',        ac_type='active').next()
c. Loading data with sample period
   hair.load(ac_type = 'active', sample_period=60).next() 
"""

#df1 = fridge.load().next()
df2 = hair.load().next()
#df3 = coffe.load().next()
#df4 = kettle.load().next()
#df5 = washing.load().next()
#df6 = computer.load().next()
#df7 = freezer.load().next()

#print df1.head()
print df2.head()
#print df3.head()
#print df4.head()
#print df5.head()
#print df6.head()
#print df7.head()

series1 = hair.load(ac_type = 'active', sample_period=60).next()
print series1.head()

###############################################################################################
# 5. MeterGroup, ElecMeter, selection and basic statistics ####################################
###############################################################################################

"""
Description for Meter Group : 
a. Classified only by submeter
   elec.submeters()
b. Calculating total energy for elecmaterID
   elec.mains().total_energy()
c. Calculating total energy for submeter (per appliance)
   energy_per_meter = elec.submeters().energy_per_meter()
d. MeterGroup which only contains the ElecMeters which used more than 100 kWh:
   
"""

print elec.submeters()
print elec.mains().total_energy()
energy_per_meter = elec.submeters().energy_per_meter()
print energy_per_meter


