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
[center_lane_driving]: ./examples/center-lane-driving.jpg "Center Lane Driving"
[backtrack_driving]: ./examples/backtrack-driving.jpg "Backtract Driving"
[recovery_from_left_1]: ./examples/recover-from-left1.jpg "Recovery from Left side 1"
[recovery_from_left_2]: ./examples/recover-from-left2.jpg "Recovery from Left side 2"
[recovery_from_right_1]: ./examples/recover-from-right1.jpg "Recovery from Right side 1"
[recovery_from_right_2]: ./examples/recover-from-right2.jpg "Recovery from Right side 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### File Submission & Code Quality

#### 1. File Submission

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Usable and readable code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

The model based on Nvidia architecture. It comprises of:
* Lambda layer for data normalization using
```
Normalized x = x / 255 - 0.5
```
* Cropping2D layer, to crop from 70 pixels from top, 25 pixels from bottom
* 3 Convolution layers, each with 24, 36, 48 depths, 5x5 filter size, stride 2x2, and RELU activation (code line 39-41)
* 2 Convolution layers, depths 64,  3x3 filter size, with RELU activation (code line 40-41)
* Flatten layer
* Dropout layer with dropout rate 0.4
* 4 Fully connected layers


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

The overall strategy for deriving a model architecture was to add training data from different road side markers, e.g. red-white, fence, soil, shadow, etc.

My first step was to use a convolution neural network model based on NVIDIA neural network. I thought this model might be appropriate because NVIDIA has tested it at real car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a Dropout layer after Flatten layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded some training at distinctive locations. 

I also added many recovery record from slightly left/right side, from far left, far right road, at every distinctive locations.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 36-52) consisted of following layers and layer sizes:
* Cropping2D layer: to crop from 70 pixels from top, 25 pixels from bottom
* Convolution layer: 24 depth, 5x5 filter size,  stride 2x2, and RELU activation 
* Convolution layers: 36 depth, 5x5 filter size,  stride 2x2, and RELU activation 
* Convolution layers: 48 depth, 5x5 filter size,  stride 2x2, and RELU activation 
* Convolution layers: 64 depths,  3x3 filter size, with RELU activation
* Flatten layer
* Dropout layer with dropout rate 0.4
* Fully connected layer 100 neurons
* Fully connected layer 50 neurons
* Fully connected layer 10 neurons
* Fully connected layer 1 neuron

The network uses Adam optimizer with target to reduce mean square error.

I tried following approaches at image pre-processing, that don't work: 
* YUV color scale, it doesn't significantly reduce error loss
* Add extra steering wheel measurement factor for image from left camera and right camera, but not only it doesn't reduce error, but it also increase  it.
* Modify the 5th convolution layer, by using 1x1 filter, but it doesn't reduce error loss.
* Add a Dropout layer after each fully connected network layer, but it doesn't reduce error rate.  

Layer that reduce error rate are:
* Cropping2D
* Flatten


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane_driving]

I also recorded backtrack center lane driving. Here is example image of backtrack center lane driving:

![alt text][backtrack_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. 

These images show what a recovery looks like starting from left side :
![alt text][recovery_from_left_1] ![alt text][recovery_from_left_2]

These images show what a recovery looks like starting from right side :
![alt text][recovery_from_right_1] ![alt text][recovery_from_right_2]

I also trained/recorded every distinct areas, such red-white road side, bridge, traffic sign, beside lake.

I tried to augment the data, but additional data doesn't improve loss score. So i decided to stick with left camera image, and right camera image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. (line 52) 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by loss score 0.054. I used Adam optimizer so that manually training the learning rate wasn't necessary.
