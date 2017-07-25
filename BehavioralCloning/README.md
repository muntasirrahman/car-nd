# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

The BehavioralCloning folder has following files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* writeup_report.md (Report file)
* video.mp4 (a video recording of the vehicle driving autonomously)



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

### `drive.py`

It can be used with drive.py using this command:

```sh
python drive.py model.h5
```

### `video.py`

```sh
python video.py run1
```

Above command will create a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

To specify the frames per second) use --fps XX, e.g.:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
