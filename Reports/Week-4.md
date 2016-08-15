# Week 4
*5th August 2016*

## PCA-SVM Occupancy Monitoring
Based on Monday (1/8) meeting, occupancy sensing module should be configurable in terms of:
* Ratio of testing and training data
* Sampling rate. Originally, data is obtained every 1 second. Because we want to limit access (i.e. protect privacy), sampling rate should be configurable to find a safe yet still accurate prediction.
* Feature length. Since sampling rate is configurable, to ensure that feature length is divisible by the resampled data, feature length should also be configurable. Originally, this is set to 15 minutes (900 data point).

The diagram of implementation is shown in figure 1 below.

![PCA-SVM implementation diagram](../images/configurable-pca-svm.png)
    **Figure 1** *Configurable PCA-SVM implementation diagram*
    
The boxes denote processes or functions, the circles denote data (dataframe, numpy array, or single value). This is similar to diagram in week 3, except that now there are:
* Resampler before feature extractor.
* Three inputs: test ratio, sampling rate, and feature length.

**Running**

```py
$ python pca-svm-cv-conf.py --house=<house_id> --tr=<test_ratio> --sr=<sampling_rate> --fl=<feature length>
```

Notes:
* house_id can be r1, r2, or r3.
* test_ratio is decimal number, 0.0 < test_ratio < 1.0. ratio of training data is 1 - test_ratio.
* sampling_rate measured in second.
* feature_length measured in second. Feature length should be at least twice the sampling_rate.

Using Python(x,y)-2.7.10.0 in Windows 8 x64 OS. Machine is i7, 16GB RAM, 512 GB SSD. 

The result will be available in 1-5 minutes depending on the size of the dataset.

**Result**

The result is obtained using a script (runner.py) that runs pca-svm-cv-conf.py over all houses, test ratios, sampling rates, and feature lengths. It is plotted on the following diagrams (each diagram represents a house).

#### Effect of Test Ratio

![Varying the test ratio](../images/tr.png)
    **Figure 2** *Varying the test ratio*	

Varying the the size of test set on all houses does not show a linear relation between size of test set and accuracy. Some size gives a result that resembles a random guess (only a little bit higher than 50%). However, we can see that 0.6 and 0.8 test set ratio gives a pretty good prediction result for all houses in general.

#### Effect of Sampling Rate

![Varying the sampling rate](../images/sr.png)
    **Figure 3** *Varying the sampling rate*	

Based on figure 3 above, decreasing the sampling rate surprisingly improves the occupation sensing accuracy, at least compared to the original 1 Hz sampling rate. Decreasing the sampling rate even further does not necessarily reduce nor improves the accuracy for all houses in general. For example, increasing the sampling period to 5 minutes (300 seconds) gives a high accuracy prediction for house r2, but it gives a low accuracy prediction for house r3.

#### Effect of Feature Length

![Varying the feature length](../images/fl.png)
    **Figure 4** *Varying the feature length*
	
Finally, in terms of feature length, the 15 minutes window that is used by the original paper [[1](#household)] gives the best result for all houses. Furthermore, using 10 minutes or 30 minutes feature length also still gives a good result.

In conclusion, decreasing the sampling rate does not necessarily drops the occupancy sensing accuracy. Furthermore, readjusting the feature length shorter to 10 minutes or longer to 30 minutes are also possible as it does not reduce the accuracy. Finally, to get a good result, the size of test set should be larger than the training set (4:3 or 4:1).

## NILMTK (ECO datasets building 1-3)

### Implemented
Converting ECO datasets to HDF5 format: We only use csv file for 3 building according to ECO dataset.
Varying training interval and test interval (2-6, 1-7)
Only used top 5 appliances for training. (using all appliances causing an error)
Disaggregating both using FHMM and CO algorithm with period of 1 minute
Metrics calculated: FTE (use built-in function from NILMTK) and Te (modification from NILMTK function)
Output format (timestamp, appliances used, energy per appliance) 

### Result

#### Output format example
1343685600000000000 Washing machine 129.0, Computer 171.0, 
1343685660000000000 Washing machine 129.0, Computer 171.0, 

#### Metrics calculated
Building 1, train 2 months, test 6 months
FTE_fhmm = 0.74096360814751006
FTE_co = 0.73125783401527555
Te_fhmm = 1.9721241761914172
Te_co = 1.5663349820381891

Building 1, train 1 month, test 7 months
FTE_fhmm = 0.73493680535081674
FTE_co = 0.72045076007507558
Te_fhmm = 1.8882833816247044
Te_co = 1.5590566567407877

Building 2, train 2 months, test 6 months
FTE_fhmm = 0.76227965085182359
FTE_co = 0.7095725318473205
Te_fhmm = 0.80670435919975736
Te_co = 0.94447361440174971

Building 2, train 1 month, test 7 months
FTE_fhmm = 0.77039752219767776
FTE_co = 0.72828439432525349
Te_fhmm = 0.91964341471141808
Te_co = 0.90876522738746557

### Not Implemented :
Metrics : Number of appliances identified correctly (Ja) and Number of appliance states identified correctly (Js)

### Problems:
Got an error while trying to use all the appliances for training process. (error message, ValueError: Must pass DataFrame with boolean values only)
Error when training for building 3. (error message, ValueError: Must pass DataFrame with boolean values only
We still confused when trying to implement Ja and Js Metrics in FHMM and CO disaggregation. We need more detail explanation of the metrics formula.

### References
1. <div id="household"/> Kleiminger, W., Beckel, C., & Santini, S. (2015). Household Occupancy Monitoring Using Electricity Meters. ETH Zurich.

 
