# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[neural_network]: ./examples/nvidia_neural_network.png "Model Architecture"
[center_lane_driving]: ./examples/center-lane-driving.jpg "Center Lane Driving"
[backtrack_driving]: ./examples/backtrack-driving.jpg "Backtract Driving"
[recovery_from_left_1]: ./examples/recover-from-left1.jpg "Recovery from Left side 1"
[recovery_from_left_2]: ./examples/recover-from-left2.jpg "Recovery from Left side 2"
[recovery_from_right_1]: ./examples/recover-from-right1.jpg "Recovery from Right side 1"
[recovery_from_right_2]: ./examples/recover-from-right2.jpg "Recovery from Right side 2"


## Rubric Points

### File Submission & Code Quality

#### 1. File Submission

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video file captured by center camera

A simulator video of autonomous driving is  at youtube http://www.youtube.com/watch?v=zmdRH4WHXh8

[![Simulator Video](http://img.youtube.com/vi/zmdRH4WHXh8/0.jpg)](http://www.youtube.com/watch?v=zmdRH4WHXh8)


#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Usable and readable code

The model.py file contains the code for training and saving the convolution neural network. The file model.py shows the pipeline I used for training and validating the model, and it contains comments to explain that particular line.

### Model Architecture and Training Strategy

#### 1. Model architecture

The model based on Nvidia architecture. It comprises of:

| Layer    | Detail      | Remarks |
|:---------|:-----------:|:--------|
|Lambda      | ```x/255 - 0.5``` |Data normalization|
|Cropping2D  |70,25 |to crop road image 70px from top, 25px from bottom|
|Convolution |24 depth, 5x5 filter,  stride 2x2, and RELU activation|Feature identification|
|Convolution |36 depth, 5x5 filter,  stride 2x2, and RELU activation|Feature identification| 
|Convolution |48 depth, 5x5 filter,  stride 2x2, and RELU activation |Feature identification|
|Convolution |64 depths,  3x3 filter size, with RELU activation|Feature identification|
| Flatten    |   |  | 
|Dropout     |rate 0.4|Overfitting prevention|
|Fully connected |100 neurons| |
|Fully connected |50 neurons | |
|Fully connected |10 neurons | |
|Fully connected |1 neuron   | |

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 45). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, from 3 different cameras (left, center, right) (code line 19-25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 52).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was: 
* by providing useful training data
  * from different type of road side markers, e.g. red-white, fence, soil, shadow from different type of road side markers, e.g. red-white, fence, soil, shadow
  * many recovery record from slightly left/right side, from far left, far right road, at every distinctive locations
* image data pre-processing
* prevent overfitting

My first step was to use a convolution neural network model based on NVIDIA neural network. I thought this model works because Nvidia has tested it and it works.

To combat the overfitting, I added a Dropout layer after Flatten layer.

The final step was to run the simulator to see how well the car was driving around track one. At the initial version, there were a few spots where the vehicle fell off the track. To fix these cases, I recorded some training data from those spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 36-52) consisted of following layers and layer sizes:
![alt text][neural_network]


The network uses Adam optimizer to reduce mean square error.

I tried following approaches at image pre-processing, but it does not work:
* YUV color scale, it does not reduce error, does not work. 
* Adding extra steering wheel measurement factor:
  * for image from left camera and right camera.
  * The extra measurement factor increase error.

I add a Dropout layer with rate 0.4 which does reduce error.

I also tried to tune the neural network by:
* Modify the 5th convolution layer
  * by using 1x1 filter, but it does not work.
* Insert a dropout layer, in between each Dense layer, but it does not reduce error rate.  


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving][center_lane_driving]

I also recorded backtrack center lane driving. Here is example image of backtrack center lane driving:

![Backtrack driving][backtrack_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. 

These images show what a recovery looks like starting from left side :

![Recovery][recovery_from_left_1] ![Recovery][recovery_from_left_2]

These images show what a recovery looks like starting from right side :

![Recovery][recovery_from_right_1] ![Recovery][recovery_from_right_2]

I also trained/recorded every distinct areas, such red-white road side, bridge, traffic sign, beside lake.

I tried to augment the data, but additional data doesn't improve error score, it even increase the error rate. So i decided to stick with left camera image, and right camera image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. (line 52) 

I used the training data for training and validation of the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by error rate 0.054. I used Adam optimizer so that manually training the learning rate wasn't necessary.
