# Week 9
*13 September 2016*

## PCA-SVM Occupancy Sensing
### Sensing accuracy
Trying to improve accuracy graph by:
* Use more training data than testing data (20:80 or 10:90 rule)
* Work only with weekdays data
* Work only with weekends data
* Train with 1 month data, test on 1 week data

All results are obtained with 10 fold cross validation.

#### 20:80 or 10:90 Rule

![All week with 10% testing](../images/week9/test_10.PNG)<br>
    **Figure 1** *All week with 10% testing*

![All week with 20% testing](../images/week9/test_20.PNG)<br>
    **Figure 2** *All week with 20% testing*

![All week with 60% testing](../images/week9/test_60.PNG)<br>
    **Figure 3** *All week with 60% testing*	

#### Weekdays data
	
![Weekdays with 10% testing](../images/week9/wd_10.PNG)<br>
    **Figure 4** *Weekdays with 10% testing*

![Weekdays with 20% testing](../images/week9/wd_20.PNG)<br>
    **Figure 5** *Weekdays with 20% testing*

![Weekdays with 60% testing](../images/week9/wd_60.PNG)<br>
    **Figure 6** *Weekdays with 60% testing*	

#### Weekends data	
	
![Weekends with 10% testing](../images/week9/we_10.PNG)<br>
    **Figure 7** *Weekends with 10% testing*

![Weekends with 20% testing](../images/week9/we_20.PNG)<br>
    **Figure 8** *Weekends with 20% testing*

![Weekends with 60% testing](../images/week9/we_60.PNG)<br>
    **Figure 9** *Weekends with 60% testing*	

#### Train with 1 month data, test on 1 week data

![Train using June, test using a week in August](../images/week9/trjun_teaug.PNG)<br>
    **Figure 10** *Train using June, test using a week in August*
	
![Train using July, test using a week in August](../images/week9/trjul_teaug.PNG)<br>
    **Figure 11** *Train using July, test using a week in August*

![Train using some weeks in August, test using a week in August](../images/week9/traug_teaug.PNG)<br>
    **Figure 12** *Train using some weeks in August, test using a week in August*