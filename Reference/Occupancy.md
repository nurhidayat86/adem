# Occupancy Detection Using Aggregated Energy Data
*based on [1], [3], and [4]*

Occupancy detection uses aggregated energy data with 1Hz sampling frequency can give detection accuracy between 84-93%. It uses 35 energy features (onoff, stdev, min, max, etc). Several classification techniques are used (svm, thresholding, gmm, hmm) and feature/dimensionality reduction such as SFS and PCA. Based on dimensionality reductions, features that shows appliance state changes are best suited to detect occupancy. Result shows that PCA-SVM method gives the highest accuracy (85% in a household).

In contrast with NILM approaches which require calibration of appliances, this approach only require users to annotate the occupancy state of the household. This can even be reduced by proposing a possible ground truth to user based on a simple heuristic occupancy detection.

However, the accuracy is low in the other households with high actual occupancy. This shows that while occupancy sensing based on energy data is promising in less occupied houses, it is not effective in highly occupied houses. Furthermore, electricity consumption is less indicative of occupancy in night time. This approach is only applied to day time. Therefore, this approach is not suitable for a smart heating in such scenario (highly occupied home, night time operation). Improving the performance could be done by fusing electricity consumption data with other sensory data.

# Inferring User Location using WiFi Trace
*based on [2]*

Sensing accuracy reaches 80% using homeset algorithm. The algorithm uses unlabelled WiFi trace data which makes it applicable in real world scenario.

The algorithm aims to create a probability matrix of 7 days in a week x N time slots.

On each time slot, a tuple consisting the timestamp and the MAC address of detected access points is stored. The algorithm will identify these stored tuples against a set of access points in the house area (known as the homeset). If the stored tuples have one or several access points in the homeset, it will increment the corresponding cell in the occupancy frequency matrix (O) by 1. Furthermore on each scan, the corresponding cell in the total observation matrix (T) is also incremented by one. Finally the probability matrix (P) is computed as Oi,j/Ti,j

To increase the reliability and eliminate manual effort to annotate the data, the homeset algorithm will perform scan at 3am to 4am to obtain the homeset, assuming user is at home at those time.

The probability matrix is considered as mature when at least 95% of the slots have a minimum of 1 observation/scan.

Using homeset algorithm, it is possible to annotate an unlabeled wifi trace dataset. This is very useful for occupancy prediction.

# Occupancy Prediction
*based on [4]* 

There are two types of occupancy prediction: binary or level. While occupancy level is useful for commercial building, residential building only requires the binary occupancy detection. To achieve this, several approaches are available: context aware (based on current data), schedule (based on historical data), and hybrid. Most algorithm such as Preheat, Presence Probabilities, and Smart Thermostat.

These algorithms are evaluated against a labeled LDCC dataset (labeled using the homeset algorithm). The accuracy is measured by comparing the prediction result and homeset. The PP (PPS) gives the best accuracy and RoC curves. PH algorithm also shows high accuracy which means that it may gives a good result for a certain home.

Finally while the result is promising, there is a limitation to the schedule based prediction which can only be improved up to 90%. The alternative is to use hybrid approach that will incorporate context aware data.

# Energy Saving
*based on [4]* 

In order to evaluate the potential energy saving, a building model must be used. There are several models, for example the resistance capacitance model. The dissertation used 5R1C based model that is reproducible. This model considers differentiation between indoor air temperature and building parts temperature, ventilation losses, solar gain, and internal gain

This simulation required historical weather and annual weather model to create weather scenario. Furthermore building heat characteristic data, ventiallation losses, and solar gains must also be calculated using the provided equation.

Finally, a smart thermostat controller implements the occupancy prediction. It calculates when to turn on and turn off heater based on occupancy dtection and prediction and also targeted heat level (comfort, setback).

However, some simplification and assumption is made, such as no relation between weather and occupancy, simple building configuration, and air temperature calculation of the 5R1C model.

# Saving Evaluation
*based on [4]* 

The saving result is measured in terms of discomfort and efficiency gain. The lost of comfort is actually lower than the misprediction probability, which is good. In terms of efficiency gain, the saving ranges from 6-17%. Maximum gain is obtained from poorly insulated building. Furthermore, it will save more energy to forgo heating if occupancy period is low. In such cases where occupant prefer to turn on heater, a simple override button on the thermostat can be a solution.

#Reference
1. Kleiminger, W., Beckel, C., Staake, T., & Santini, S. (2013). Occupancy Detection from Electricity Consumption Data. Proceedings of the 5th ACM Workshop on Embedded Systems For Energy-Efficient Buildings - BuildSys'13.
2. Kleiminger, W., Beckel, C., Dey, A., & Santini, S. (2013). Inferring Household Occupancy Patterns from Unlabelled Sensor Data. ETH Zurich.
3. Kleiminger, W., Beckel, C., & Santini, S. (2015). Household Occupancy Monitoring Using Electricity Meters. ETH Zurich.
4. Kleiminger, W. (2015). Occupancy Sensing and Predictionfor Automated Energy Savings. ETH Zurich.
