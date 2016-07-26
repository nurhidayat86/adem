# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 02:02:49 2016

@author: neo
"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')
from nilmtk import DataSet

#Importing elec data
iawe = DataSet('/home/neo/iAWE/iawe.h5')

#Save electrical data in building[1] into elec variable.
elec = iawe.buildings[1].elec
elecmain = elec.mains()
elecsub = elec.submeters()

#printing summed power series in main group
print("Summed main elecgroup")
print(elecmain.power_series_all_data().head())
print("Summed sub elecgroup")
print(elecsub.power_series_all_data().head())

#proportion energy
elecprop = elec.proportion_of_energy_submetered()
print("Printing proportion energy")
print(elecprop)

#load
print("Printing load")
print(elec.load())