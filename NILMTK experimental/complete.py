# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 02:02:49 2016

@author: neo
"""

from nilmtk.utils import print_dict
from nilmtk import DataSet


iawe = DataSet('/home/neo/iAWE/iawe.h5')
elec = iawe.buildings[1].elec
print("---Available Structure electric data iawe")
print (elec)
print("")
fridge = elec['fridge']
smart = elec[2]
print("---Available columns smart")
print (smart.available_columns())
print("")
print("Smart")
smt = smart.load().next()
print(smt.head())
print("")
print("---Available columns fridge")
print (fridge.available_columns())
print("")
print("---Loading all fridsge columns save to df---")
df1 = fridge.load().next()
print(df1.head())
