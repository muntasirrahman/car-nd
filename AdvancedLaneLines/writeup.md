## **Advanced Lane Line**

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_chessboard]: output_images/img_chessboard.png "Chessboard Calibration"
[img_undist2]: output_images/img_undist2.png "Undistorted"

[video1]: ./project_video.mp4 "Video"
[video_output]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This is the writeup page.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "AdvancedLaneLines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][img_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here the examples of distortion correction to one of the test images:
![alt text][img_undist2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color transform and Sobel X gradient thresholds to generate a binary image. The Sobel X operator is to find white/black pixels. The saturation channel is to find yellow/white pixels on gray area. The lightness channel is to filter out shadow image. Then those filters are combined to get the final filter.

```python
img_filter[((img_lgh == 1) & (img_sat == 1) | (img_sobel == 1))] = 1
```

The code for color tranform is at at 5th code cell in function `color_transform()`. 

Here's an output example of this step. 

[img_color_transform]: output_images/img_color_transform.png "Color and Threshold Transform"

![alt text][img_color_transform]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Before an image is transformed, the image is cropped in some region at 6th code cell  at function `crop()`. The function `crop()` purpose is to clean noise pixel outside and inside certain region.

[img_crop]: output_images/img_crop.png "Cropping region"
 ![alt text][img_crop]

Gradien `g_M` variable, gradient inverse variable `g_M_inverse`, and rectangle points `g_rect_points` are pre-calculated once at 7th code cell at function `get_transform_m()`, to be used later within pipeline function`. 

The code for perspective transform in at function `warp()` in the 8th code cell of the IPython notebook). The warp function use `cv2.warpPerspective()` function to get bird eye view. 

I chose the hardcode the source and destination points in the following manner:

```python
    y_t = 457
    y_b = 720

    x_tl = 586  # x_tl + 6
    x_tr = 692  # x_tr - 8

    x_bl = 220  # x_bl + 30
    x_br = 1100  # x_br - 30

    top_left_src = np.array([x_tl, y_t])
    top_right_src = np.array([x_tr, y_t])

    bottom_left_src = np.array([x_bl, y_b])
    bottom_right_src = np.array([x_br, y_b])

    offset = (x_tl - x_tr + x_br - x_bl) // 4

    top_left_dst = np.array([x_tl - offset, 0])
    top_right_dst = np.array([x_tr + offset, 0])

    bottom_left_dst = np.array([x_bl + offset, y_b])
    bottom_right_dst = np.array([x_br + - offset, y_b])

    reg_src = np.float32([top_left_src, top_right_src, bottom_right_src, bottom_left_src])
    reg_dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 586, 457      | 393, 0        | 
| 692, 457      | 885, 0        |
| 1100, 720     | 907, 720      |
| 220, 720      | 413, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[img_warp]: output_images/img_warp.png "Perspective transform"

![alt text][img_warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function `line_detection()` at 8th code cell, finds lane-line pixels using maximum value of histogram. Nine sliding windows find good pixels, then add good pixels to lane indices array. 

Those pixels are fitted into polynomial function

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
```

[img_polyfit]: output_images/img_polyfit.png "Polynomial fit of lane line pixels"
![alt text][img_polyfit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The 10th code cell does curvature radius calculation at function `calc_radius_curvature()`.

Radius curvature i calculated by using `np.polyfit()` of converted line pixels. 

```python
y_eval = np.max(ploty)

left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
``` 

The car position in the lane is calculated by differences between the center of lane, and the center of images. The difference in pixel then converted to meter.

```python
# Calculate center
center_dist_in_px = (1280 - rightx[700] - leftx[700]) / 2
center_dist = center_dist_in_px * xm_per_pix
``` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `render_lane_lines()`.  Here is an example of my result on a test image:

[img_result]: output_images/img_rendered.png
![alt_text][img_result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I use HLS color space and Sobel X operator to find the lane lines. I presume the lane line is always within certain region of camera, which is used for cropping function. The crop function crops both outside and inside region. The inside crop function is used to reduce noise pixels that will disturb polynomial fit function. This approach will cause trouble when lane line does not fit to those regions.

I should use different threshold and color space to filter out black markers on road instead of inside cropping approach. I haven't found it because but i have run out of time.  
