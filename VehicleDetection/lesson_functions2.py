import numpy as np
import cv2
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from random import randint

# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  block_norm='L1',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       block_norm='L1',
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(16, 16)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=16, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='YCrCb', spatial_size=(16, 16),
                     hist_bins=16, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def single_img_features(img, color_space='YCrCb', spatial_size=(16, 16),
                        hist_bins=16, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='YCrCb',
                   spatial_size=(16, 16), hist_bins=16,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        # prediction = clf.predict(test_features)
        decision = clf.decision_function(test_features)
        if decision > 0.6:
            prediction = 1
        else:
            prediction = 0

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


def show_image(image_list, label_list=None, cols=2, fig_w=14, fig_h=6, show_axis='on', debug=False, cmap='gray'):
    if label_list is None:
        label_list = []
        label_found = False
    else:
        label_found = True

    n_img = len(image_list)
    if n_img < 1 and debug:
        print('No image to be shown'.format(n_img))
        pass

    rows = n_img // cols
    if rows == 0:
        rows = 1
    if rows * cols < n_img:
        rows += 1

    gs = gridspec.GridSpec(rows, cols)
    plt.close('all')
    plt.figure(figsize=(fig_w, fig_h))

    for i, img in enumerate(image_list):
        fig = plt.subplot(gs[i])
        if label_found:
            fig.set_title(label_list[i], fontsize=12)

        fig.imshow(img, cmap=cmap)
        fig.axis(show_axis)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.)
    plt.show()


def capture_frame_from_video(video_file_name):
    vid_capture = cv2.VideoCapture(video_file_name)
    status, image = vid_capture.read()
    count = 0
    while status:
        cv2.imwrite('video_cap/frame_{:04d}.jpg'.format(count), image)
        status, image = vid_capture.read()
        count += 1

def convert_color(img, color_space, debug=False):

    if color_space != 'RGB':
        if color_space == 'HSV':
            img_result = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        elif color_space == 'LUV':
            img_result = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

        elif color_space == 'HLS':
            img_result = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        elif color_space == 'YUV':
            img_result = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        elif color_space == 'YCrCb':
            img_result = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    else:
        img_result = np.copy(img)

    if debug:
        max_src = np.amax(img)
        max_result = np.amax(img_result)
        print('Data type:{}, Color: {}. Max:{:.3f}, {:.3f}'.format(img.dtype, color_space, max_src, max_result))
        print('Image source:', img[:1, :3, 0])
        print('Image result:', img_result[:1, :3, 0])


    return img_result


def extract_single_image_features(img_src,
                                  color_space,
                                  spatial_size=(16, 16),
                                  hist_bins=32,
                                  orient=9,
                                  pix_per_cell=8,
                                  cell_per_block=2,
                                  hog_channel=0,
                                  spatial_feat=True, hist_feat=True, hog_feat=True,
                                  debug=False):
    features = []
    img = img_src.astype(np.float32)

    #Detect PNG format
    max_val = np.amax(img)
    if max_val <= 1:
        if debug:
            print('Convert image format from PNG')
        img = img * 255

    img_feat = convert_color(img, color_space)

    if spatial_feat:
        spatial_features = bin_spatial(img_feat, size=spatial_size)
        features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(img_feat, nbins=hist_bins)
        features.append(hist_features)


    img_hog = None
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img_feat.shape[2]):

                hog_features.extend(get_hog_features(img_feat[:, :, channel],
                                                     orient=orient,
                                                     pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block,
                                                     vis=False,
                                                     feature_vec=True))
        else:

            hog_features, img_hog = get_hog_features(img_feat[:, :, hog_channel],
                                                     orient=orient,
                                                     pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block,
                                                     vis=True,
                                                     feature_vec=True)

        features.append(hog_features)

    if debug:
        return np.concatenate(features), img_hog
    else:
        return np.concatenate(features)


def show_array(val_list, lbl_list=None, cols=2, fig_w=10, fig_h=10):
    # Plot an example of raw and scaled features

    n = len(val_list)
    rows = n // cols
    if rows == 0:
        rows = 1
    if rows * cols < n:
        rows += 1
    gs = gridspec.GridSpec(rows, cols)
    plt.close('all')
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('off')

    for i, data in enumerate(val_list):
        fig = plt.subplot(gs[i])
        n_dim = data.ndim
        if n_dim == 1:
            fig.axis('on')
            fig.plot(data)
        else:
            fig.axis('off')
            fig.imshow(data)

        if lbl_list is not None:
            fig.set_title(lbl_list[i])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.)
    plt.show()


def view_sample(car_path_list, notcar_path_list, color_space, hog_channel):
    chart_list = []
    lbl_list = []
    list_name = 'Car'

    for path in [car_path_list[randint(0, len(car_path_list))], notcar_path_list[randint(0, len(notcar_path_list))]]:
        img_png = mpimg.imread(path)
        chart_list.append(img_png)
        lbl_list.append(list_name + ' RGB')

        img_conv = convert_color(img_png, color_space, debug=True) * 255
        chart_list.append(img_conv[:, :, hog_channel].ravel())
        # chart_list.append(img_conv)
        lbl_list.append(list_name + ' ' + color_space)

        feat_spatial, _ = extract_single_image_features(img_png, color_space, hist_feat=False, hog_feat=False, debug=True)
        chart_list.append(feat_spatial)
        lbl_list.append(list_name + ' Spatial')

        feat_hist, _ = extract_single_image_features(img_png, color_space, spatial_feat=False, hog_feat=False, debug=True)
        chart_list.append(feat_hist)
        lbl_list.append(list_name + ' Color Hist')

        feat_hog, img_hog = extract_single_image_features(img_png, color_space, spatial_feat=False, hist_feat=False, debug=True)
        chart_list.append(feat_hog)
        lbl_list.append(list_name + ' HOG')

        chart_list.append(img_hog)
        lbl_list.append(list_name + ' HOG Image')

        features = extract_single_image_features(img_png, color_space, hog_channel=hog_channel, debug=False)
        chart_list.append(features)
        lbl_list.append(list_name + ' Combined')

        list_name = 'Not Car'

    show_array(chart_list, lbl_list, cols=7, fig_w=18, fig_h=6)
