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

![r1 Occupancy Sensing Accuracy](../images/r1-configurable-accuracy.png)
    **Figure 2** *r1 Occupancy sensing accuracy*
	
![r2 Occupancy Sensing Accuracy](../images/r2-configurable-accuracy.png)
    **Figure 3** *r2 Occupancy sensing accuracy*	
	
![r3 Occupancy Sensing Accuracy](../images/r3-configurable-accuracy.png)
    **Figure 4** *r3 Occupancy sensing accuracy*