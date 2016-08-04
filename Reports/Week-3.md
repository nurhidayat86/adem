# Week 3
*29th July 2016*

We are implementing the PCA-SVM Household Occupancy Monitoring algorithm as explained in [[1](#household)]. Furthermore, we are exploring NILMTK. Both of our works use the ECO dataset [[2](#eco)].

## PCA-SVM Occupancy Monitoring
The diagram of implementation is shown in figure 1 below.

![PCA-SVM implementation diagram](../images/pca-svm.png)
    **Figure 1** *PCA-SVM implementation diagram*
    
The boxes denote processes or functions, the circles denote data (dataframe, numpy array, or single value). 

Based on files inside a dataset directory, raw data are extracted. Furthermore, the date information from each file is also stored. This array of dates will be used to extract the correct ground truth label.

The 35 features are extracted from 900 data point as mentioned in [[1](#household)]. The ground truth label is extracted by majority voting from 900 ground truth data point (same as the features). Next, features and ground truth labels are split into 2 fold training and testing set (X_train, y_train, X_test, y_test) using the cross_validation module.

After that, the training set features are reduced using PCA with L components (components that give 95% variance). The reduced training set features and ground truth label are then fed into a SVM classifier. After exploring kernels and parameters using GridSearchCV, RBF kernel with a certain C and gamma parameter is used.

Finally, the testing set features are fed into this classifier, and the prediction result is obtained. Accuracy is then measured using accuracy_score (in sklearn.metrics) - comparing prediction result with the ground truth testing set.

**Running**

```py
$ python pca-svm-cv2.py
```

Using Python(x,y)-2.7.10.0 in Windows 8 x64 OS. Machine is i7, 16GB RAM, 512 GB SSD. 

The result will be available in 1-5 minutes depending on the size of the dataset.

**Result**

Using a small dataset (<30) causes the result to vary extremely, possibly due to overfitting (30% to 99% accuracy, depending on which files are included in the dataset). 

After carefully adding most of the dataset (currently only tested on the summer days; several csv files have negative energy values (-1,-1,-1) and as of now are removed from the dataset) and increase the size to 50-60 days, the accuracy ranges from 80% to 95% which is similar as the result in [[1](#household)].

**Improvement**
1. Performing kernel trick as mentioned in [[1](#household)]. We wanted to apply kernel trick that will allow us to bring the features to a higher dimensional space [[5](#trick)], but we are unsure how to derive such kernel equation.
2. Run in other houses (currently only on house #2) and other season (winter)


##NILMTK iAWE dataset

For the initial phase, we are exploring the NILMTK [[6](#nilmtk)] with iAWE (Indian Dataset for Ambient Water and Energy) [[7](#iawe)] dataset. We follow all the instruction and description from the NILMTK user guide. Our expectation from exploring NILMTK in the initial phase is the result from the NILMTK user guide [[8](#guide)] can be replicated by us. Mostly, in the NILMTK is used REDD [[9](#redd)] dataset. However, the REDD dataset needs account to download their dataset, we have sent an email to the owner of REDD dataset, but they have not replied our email. As a replacement for REDD dataset, we use iAWE dataset because iAWE dataset is used in the user guide (Loading data into memory section) so we cannot replicate all the process in the NILMTK github. There are some errors (the list of error shown in folder NILMTK/errormessage.txt) and the output is not exactly same with the result from NILMTK user guide [[8](#guide)]. Although we cannot produce the pricesly output compared to the NILMTK user guide because of the difference dataset, we still can generate the main features from NILMTK such as : 

* Open HDF5 in NILMTK.
* Loading data into memory : Load all columns (default), Load a single column of power data, Specify physical_quantity or AC type, Loading by specifying AC type, Loading by resampling to a specified period   
* MeterGroup, ElecMeter, selection and basic statistics : Proportion of energy submetered, 
Active, apparent and reactive power, Total Energy, Energy per submeter, 
Select meters on the basis of their energy consumption, Stats and info for individual meters, Get upstream meter, Metadata about the class of meter, Dominant appliance, Total energy,  Dropout rate, Select subgroups of meters, Select a group of meters from properties of the meters (not the appliances), Select a single meter from a MeterGroup, Search for a meter using appliances connected to each meter, Search for a meter using details of the ElecMeter, Instance numbering, Select nested MeterGroup
* Processing pipeline, preprocessing and more stats : Load a restricted window of data,  The Apply preprocessing node,  Fill gaps in appliance data

##Additional code when using iAWE dataset (__init__.py (../nilmtk/nilmtk/dataset_converters)

```py
from .iawe.convert_iawe import convert_iawe  
```
### References
1. <div id="household"/> Kleiminger, W., Beckel, C., & Santini, S. (2015). Household Occupancy Monitoring Using Electricity Meters. ETH Zurich.
2. <div id="eco"/> [ECO dataset](https://hazelcast.com/products/)
3. <div id="occupancy"/> Kleiminger, W. (2015). Occupancy Sensing and Prediction for Automated Energy Savings. ETH Zurich.
4. <div id="buildsys"/> Kleiminger, W., Beckel, C., Staake, T., & Santini, S. (2013). Occupancy Detection from Electricity Consumption Data. Proceedings of the 5th ACM Workshop on Embedded Systems For Energy Efficient Buildings Build Sys '13
5. <div id="trick"/> http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html, last accessed on 28th July 2016.
6. <div id="nilmtk"/> Nipun Batra, Jack Kelly, Oliver Parson, Haimonti Dutta, William Knottenbelt, Alex Rogers, Amarjeet Singh, Mani Srivastava. NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring. In: 5th International Conference on Future Energy Systems (ACM e-Energy), Cambridge, UK. 2014.
7. <div id="iawe"/> http://iawe.github.io/
8. <div id="guide"/> https://github.com/nilmtk/nilmtk/tree/master/docs/manual
9. <div id="redd"/> http://redd.csail.mit.edu/