# Week 7
*3 September 2016*

## PCA-SVM Occupancy Sensing
### Extra information
Based on Monday (1/8) meeting, occupancy sensing module should be able to predict:
* Room level occupancy
* Minimum number of occupants in the house

Using the output of house level occupancy prediction and the output of NILMTK algorithm, room level occupancy prediction and minimum number of occupant in the house is predicted. The step of room level occupancy detection is seen as figure bellow:

![Expanded occupancy sensing](../images/room level occupancy.png)<br>
    **Figure 1** *Expanded occupancy sensing*

The rules library consists of if - then rules that detects the existance of user based on the disaggregated appliance data: The appliance that produces high energy demand indicates the user activity in the place where the appliance is located.

Using Orange data mining library, we can obtain association rules. For example, given a set of appliances, we can have a certain confidence that a certain appliance is also used together.

Rules shown in figure 2 below is derived from a basket file that is based on house r2 ground truth data (plugs data) in the ECO dataset (support 0.2, confidence 0.6):

![Basket rules of active appliances](../images/rule-02-06.PNG)<br>
    **Figure 2** *Basket rules of active appliances*

We have calculate the probability of other appliances ON/OFF states when one particular appliance is in the ON state, as bellow:

![Chart 0.4 test ratio](../images/images/appliances_state_probability/AC.png)<br>
    **Figure 3**

![Chart 0.5 test ratio](../images/images/appliances_state_probability/Audio.png)<br>
    **Figure 4**
	
![Chart 0.6 test ratio](../images/images/appliances_state_probability/Kettle.png)<br>
    **Figure 5**
	
![Chart 0.7 test ratio](../images/images/appliances_state_probability/TV.png)<br>
    **Figure 6**
	
![Chart 0.8 test ratio](../images/images/appliances_state_probability/dishwahser.png)<br>
    **Figure 7**

![Chart 0.4 test ratio](../images/images/appliances_state_probability/freezer.png)<br>
    **Figure 8**

![Chart 0.5 test ratio](../images/images/appliances_state_probability/fridge.png)<br>
    **Figure 9**
	
![Chart 0.6 test ratio](../images/images/appliances_state_probability/htpc.png)<br>
    **Figure 10**
	
![Chart 0.7 test ratio](../images/images/appliances_state_probability/lamp.png)<br>
    **Figure 11**
	
![Chart 0.8 test ratio](../images/images/appliances_state_probability/laptop_computer.png)<br>
    **Figure 12**
    
![Chart 0.6 test ratio](../images/images/appliances_state_probability/stove.png)<br>
    **Figure 13**
	
![Chart 0.7 test ratio](../images/images/appliances_state_probability/tablet_charger.png)<br>
    **Figure 14**
	
![Chart 0.8 test ratio](../images/images/appliances_state_probability/television.png)<br>
    **Figure 15**

### Sensing accuracy
After some modification on how to compute features (e.g. operates directly on single dataframe) we can see that several testing scenarios produced a desired output, i.e. as sampling rate is reduced, accuracy drops. One thing to note is that the ETHZ paper used sampling rate 1 seconds and feature length/labeling period 900 seconds. See these test scenarios:
* Test ratio 0.4, feature length (in second) 300, 600, and 1800
* Test ratio 0.5, feature length (in second) 3600
* Test ratio 0.6, feature length (in second) 600, 900, and 3600
* Test ratio 0.7, feature length (in second) 1800 and 3600
* Test ratio 0.8, feature length (in second) 300, 600, 1800, and 3600

See the following charts for more detail:

![Chart 0.4 test ratio](../images/acc-04.png)<br>
    **Figure 16** *Accuracy vs labeling period with test ratio 0.4*

![Chart 0.5 test ratio](../images/acc-05.png)<br>
    **Figure 17** *Accuracy vs labeling period with test ratio 0.5*
	
![Chart 0.6 test ratio](../images/acc-06.png)<br>
    **Figure 18** *Accuracy vs labeling period with test ratio 0.6*
	
![Chart 0.7 test ratio](../images/acc-07.png)<br>
    **Figure 19** *Accuracy vs labeling period with test ratio 0.7*
	
![Chart 0.8 test ratio](../images/acc-08.png)<br>
    **Figure 20** *Accuracy vs labeling period with test ratio 0.8*

Tables below are provided to show even more detail:

![Table 0.5 test ratio](../images/tacc-05.JPG)<br>
	**Table 1** *Accuracy vs labeling period with test ratio 0.5*
	
![Table 0.6 test ratio](../images/tacc-06.JPG)<br>
	**Table 2** *Accuracy vs labeling period with test ratio 0.6*
	
![Table 0.7 test ratio](../images/tacc-07.JPG)<br>
	**Table 3** *Accuracy vs labeling period with test ratio 0.7*    
