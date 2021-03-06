# Week 9
*13 September 2016*



## PCA-SVM Occupancy Sensing
### Sensing accuracy

UPDATE 15/9. Calculate tp, fp, tn, fn on the 1 month training and 1 week testing scenario. We found that:
1. When training is done using June data, 90% of errors are caused by false positives
2. When training is done using August data, 90% of errors are caused by false negatives
3. When training is done using July data, 95% of errors are caused by false positives

![TP FP TN FN](../images/week9/tp_until_fn.png)<br>
    **Figure 1** *TP FP TN FN*

![Precision with June training](../images/week9/jun_rec.png)<br>
    **Figure 1** *Precision with June training*

![Recall with June training](../images/week9/jun_prec.png)<br>
    **Figure 2** *Recall with June training*

![F with July training](../images/week9/jul_f.png)<br>
    **Figure 4** *F values with July training*

![Precision with July training](../images/week9/jul_rec.png)<br>
    **Figure 5** *Precision with July training*

![Recall with July training](../images/week9/jul_prec.png)<br>
    **Figure 6** *Recall with July training*

![F with June training](../images/week9/jul_f.png)<br>
    **Figure 7** *F values with July training*
	
![Precision with August training](../images/week9/aug_rec.png)<br>
    **Figure 8** *Precision with August training*

![Recall with August training](../images/week9/aug_prec.png)<br>
    **Figure 9** *Recall with August training*

![F with August training](../images/week9/aug_f.png)<br>
    **Figure 10** *F values with August training*	

Trying to improve accuracy graph by:<br>
1. [Use more training data than testing data (20:80 or 10:90 rule)](#twentyeighty)<br>
2. [Work only with weekdays data](#weekdays)<br> 
3. [Work only with weekends data](#weekends)<br>
4. [Train with 1 month data, test on 1 week data](#onemonth)<br>

All results are obtained with 10 fold cross validation.

#### 20:80 OR 10:90 RULE <a name="twentyeighty"></a>

![All week with 10% testing](../images/week9/test_10.PNG)<br>
    **Figure 11** *All week with 10% testing*

![All week with 20% testing](../images/week9/test_20.PNG)<br>
    **Figure 12** *All week with 20% testing*

![All week with 60% testing](../images/week9/test_60.PNG)<br>
    **Figure 13** *All week with 60% testing*	

#### WEEKDAYS DATA <a name="weekdays"></a>
	
![Weekdays with 10% testing](../images/week9/wd_10.PNG)<br>
    **Figure 14** *Weekdays with 10% testing*

![Weekdays with 20% testing](../images/week9/wd_20.PNG)<br>
    **Figure 15** *Weekdays with 20% testing*

![Weekdays with 60% testing](../images/week9/wd_60.PNG)<br>
    **Figure 16** *Weekdays with 60% testing*	

#### WEEKENDS DATA <a name="weekends"></a>
	
![Weekends with 10% testing](../images/week9/we_10.PNG)<br>
    **Figure 17** *Weekends with 10% testing*

![Weekends with 20% testing](../images/week9/we_20.PNG)<br>
    **Figure 18** *Weekends with 20% testing*

![Weekends with 60% testing](../images/week9/we_60.PNG)<br>
    **Figure 19** *Weekends with 60% testing*	

#### TRAIN WITH 1 MONTH DATA, TEST ON 1 WEEK DATA <a name="onemonth"></a>

![Train using June, test using a week in August](../images/week9/trjun_teaug.png)<br>
    **Figure 20** *Train using June, test using a week in August*
	
![Train using July, test using a week in August](../images/week9/trjul_teaug.png)<br>
    **Figure 21** *Train using July, test using a week in August*

![Train using some weeks in August, test using a week in August](../images/week9/traug_teaug.png)<br>
    **Figure 22** *Train using some weeks in August, test using a week in August*
    
## NILMTK
Comparing the result metrics (FTE, Te, and Ja)  of Original CO, CO centroid, CO Centroid + Priority, Original CO+Priority  between All Appliances and Top-5 Appliances
### Summary
####1.Comparing the result between Original CO vs centroid vs centroid+priority <br>

![Comparing the result between Original CO, Centroid, and Centroid+Priority Train 10-11 June Test 11-12 June and Test 12-13 June](../images/week9/2004.png)<br>
    **Figure 23** *Comparing the result between Original CO, Centroid, and Centroid+Priority Train 10-11 June Test 11-12 June and Test 12-13 June*

![Comparing the result between Original CO, Centroid, and Centroid+Priority](../images/week9/2049.png)<br>
    **Figure 24** *Comparing the result between Original CO, Centroid, and Centroid+Priority*

####2. Comparing the result between Original CO and Original CO + CO priority
![Comparing the result between Original CO and Original CO + CO priority Train 10-11 June Test 11-12 June and Test 12-13 June](../images/week9/1940.png)<br>
    **Figure 25** *Comparing the result between Original CO and Original CO + CO priority Train 10-11 June Test 11-12 June and Test 12-13 June* 

![Comparing the result between Original CO and Original CO + CO priority](../images/week9/2123.png)<br>
    **Figure 26** *Comparing the result between Original CO and Original CO + CO priority*

