# Week 5
*13rd August 2016*

## PCA-SVM Occupancy Monitoring
Based on Monday (1/8) meeting, occupancy sensing module should be able to predict:
* Room level occupancy
* Minimum number of occupants in the house`

The diagram of the system is shown in figure 1 below.

![Expanded occupancy sensing](../images/expanded-sensing.png)<br>
    **Figure 1** *Expanded occupancy sensing*
    
The green boxes denote functions, grey documents denote raw data from ECO dataset, and blue documents denote prediction output. 
### Not implemented
PCA-GMMHMM Occupancy monitoing. Error:
* Got an error while trying to train GMMHMM model, with function: hmm.GMMHMM(attributes).fit(x). It expect input : "2 dimension list of array like input".

### Implemented (extra)
PCA-ANN with 10 neurons, sigmoid activation function, and iteration = 200. Got 92-96% accuracy.

## NILMTK (ECO datasets building 1-3)

### Implemented
* Converting ECO datasets to HDF5 format: We only use csv file for 3 building according to ECO dataset.
* Varying training interval and test interval (2-6, 1-7)
* Only used top 5 appliances for training. (using all appliances causing an error)
* Disaggregating both using FHMM and CO algorithm with period of 1 minute
* Metrics calculated: FTE (use built-in function from NILMTK) and Te (modification from NILMTK function)
* Output format (timestamp, appliances used, energy per appliance) 

### Result

#### Output format example

Timestamp | App 1 | App 2
------------ | ------------- | -------------
1343685600000000000 | Washing machine 129.0 | Computer 171.0
1343685660000000000 | Washing machine 129.0 | Computer 171.0

#### Metrics calculated

* Building 1, train 2 months, test 6 months
    * FTE_fhmm = 0.74096360814751006
    * FTE_co = 0.73125783401527555
    * Te_fhmm = 1.9721241761914172
    * Te_co = 1.5663349820381891

* Building 1, train 1 month, test 7 months
 * FTE_fhmm = 0.73493680535081674
 * FTE_co = 0.72045076007507558
 * Te_fhmm = 1.8882833816247044
 * Te_co = 1.5590566567407877

* Building 2, train 2 months, test 6 months
 * FTE_fhmm = 0.76227965085182359
 * FTE_co = 0.7095725318473205
 * Te_fhmm = 0.80670435919975736
 * Te_co = 0.94447361440174971

* Building 2, train 1 month, test 7 months
 * FTE_fhmm = 0.77039752219767776
 * FTE_co = 0.72828439432525349
 * Te_fhmm = 0.91964341471141808
 * Te_co = 0.90876522738746557

### Not Implemented :
Metrics : Number of appliances identified correctly (Ja) and Number of appliance states identified correctly (Js)

### Problems:
* Got an error while trying to use all the appliances for training process. (error message, ValueError: Must pass DataFrame with boolean values only)
* Error when training for building 3. (error message, ValueError: Must pass DataFrame with boolean values only
* We still confused when trying to implement Ja and Js Metrics in FHMM and CO disaggregation. We need more detail explanation of the metrics formula.
