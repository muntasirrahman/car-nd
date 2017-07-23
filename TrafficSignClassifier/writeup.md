## **Traffic Sign Recognition** 

---

[//]: # (Image References)
[train_data_signs]: ./images/train_data_signs.png "Train Data Signs"
[train_data_dist]: ./images/train_data_dist.png "Train Data Distribution per Class"

[data_aug_dist]: ./images/image_data_processing.png "Image Data Processing"

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images.
* The size of the validation set is 4,410 images.
* The size of test set is 12,630 images.
* The shape of a traffic sign image is (32,32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][train_data_dist]

![alt text][train_data_signs]


### Design and Test a Model Architecture

#### 1. Data normalization methods.

[img_data_proc]: ./images/image_data_proc.png "Image Data Processing"
[aug_data_signs]: ./images/aug_train_data_signs.png "Augmented Data Before and After"
[aug_data_dist]: ./images/aug_train_data_dist.png "Augmented Data Distribution"

I explored several data normalization methods. The best method what i find is by converting to relative value from mean divided by standard deviation.

![alt text][img_data_proc]

I didn't convert image to grayscale because it doesn't improve validation accuracy.

I decided to generate additional data because it improves validation accuracy.

To add more data to the the data set, I used the following technques:
* find classes that has not sufficient data
* randomly select an image within that class
* randomly select rotation for the image (5, 10, 15 degree)
* repeat to select image step again

Here is an example of an original image and an augmented image:

![alt text][aug_data_signs]

The difference between the original data set and the augmented data set is the following
![alt text][aug_data_dist]

#### 2.Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding						|
| Max pooling			| 2x2 stride, valid padding						|
| RELU					|												|
| Convolution 5x5		| 5x5 stride, valid padding						|
| Max pooling			| 2x2 stride, valid padding						|
| Flatten				|												|
| Fully connected		| mu 0, sigma 0.1								|
| RELU					|												|
| Dropout 				| default dropout rate 0.5						|
| Fully connected		| mu 0, sigma 0.1								|
| RELU 					|												|
| Dropout 				| default dropout rate 0.5						|
| Fully connected		| mu 0, sigma 0.1								|
|						|												|
|						|												|
 


#### 3. Model Training. 
The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, based on Udacity's LeNet solution which is different thant original optimizer used by Lenet. I haven't try other optimizer, because of time constraint.

I use following parameters:
* EPOCH 14
* Batch size 64
* Learning rate 0.001

I come up with those numbers by trial and error.

#### 4. The approach taken for finding a solution and getting the validation set accuracy

My final model results were:
* training set accuracy of .975
* validation set accuracy of .974 
* test set accuracy of .960

I start with LeNet architecture template, without changing any parameter, except input size, but the accuracy result is very low. I choose that step because i have no other idea, except LeNet.

Then i changed some parameters one hot from 10 to 43, to match output result. The accuracy grows to .92.

The next steps are:
* the convolution layer 2 shape from 6 to 24
* the layer 3 output from 120 to 480
* the layer 4 output from 84 to 100
the accuracy slightly rise to 0.94

To prevent overfitting, i add some dropout layer, with dropout rate 0.5, the accuracy slightly rise to 0.96.

Then reduce the batch size from 128 to 64, increase the EPOCH to 14. The accuracy grows to 0.97.

All of those numbers, including EPOCH, Batch size, and learning rate, are found by running the code hundred of times, every day during entire week, increasing/decreasing the parameters value one by one.

I choose that iterative trial-error approach, because of time constraint. Previously i try to understand the image shape by trying to visualize of each image class in every layers will in, but it took very long time.


### Test a Model on New Images

#### 1. Five German traffic signs found on the web

[new_sign3]: ./new_signs/sign3.jpg "Sign 3"
[new_sign5]: ./new_signs/sign5.jpg "Sign 5"
[new_sign7]: ./new_signs/sign7.jpg "Sign 7"
[new_sign8]: ./new_signs/sign8.jpg "Sign 8"
[new_sign9]: ./new_signs/sign9.jpg "Sign 9"

![1 Image][new_sign8]
The image might be difficult to classify because of the image occupy all image area, and gray background.

![alt text][new_sign9]
The image might be difficult to classify because of image occupy all image area till the edge.

![alt text][new_sign3]
The image might be difficult to classify because of backround image of forrest.

![alt text][new_sign5]
The image might be difficult to classify because the image is slightly bent.

![alt text][new_sign7]
The image might be difficult to classify because the image is slightly bent.


#### 2. Model's predictions on new traffic signs and Comparison on the test set prediction. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution    	| General caution								| 
| Road work     		| Road work 									|
| Children crossing		| Children crossing								|
| No passing      		| No passing					 				|
| No vehicles			| Go straight or right    						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96%


#### 3. Prediction Result Analysis
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a General caution. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution   							| 
| 0.00     				| Speed limit (20km/h)							|
| 0.00					| Speed limit (30km/h)							|
| 0.00	      			| Speed limit (50km/h)			 				|
| 0.00				    | Speed limit (60km/h) 							|



For the second image, the model is pretty sure this is a Road work. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work										| 
| 0.00     				| Dangerous curve to the right					|
| 0.00					| No passing for vehicles over 3.5 metric tons	|
| 0.00	      			| Speed limit (80km/h)			 				|
| 0.00				    | Priority road									|



For the third image, the model is quite sure that this is a Children crossing. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.72         			| Children crossing   							| 
| 0.27     				| Beware of ice/snow							|
| 0.07					| Right-of-way at the next intersection			|
| 0.00	      			| General caution								|
| 0.00				    | Slippery road									|


For the fourth image, the model is pretty sure that this is a No passing. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing   									| 
| 0.00     				| Slippery road)								|
| 0.00					| End of no passing								|
| 0.00	      			| Dangerous curve to the left					|
| 0.00				    | Speed limit (50km/h) 							|


For the fifth image, the model incorrectly predit that this is a Speed limit (30km/h). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.77         			| Speed limit (30km/h)   						| 
| 0.15     				| Speed limit (70km/h)							|
| 0.05					| No vehicles									|
| 0.00	      			| Speed limit (50km/h)			 				|
| 0.00				    | Speed limit (80km/h) 							|

In my opinion the root cause is the image has blurry image in the center. The model confuses it with other speed limit signs. 






