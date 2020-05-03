# TO-DO: sliding point window, line generation, anti-birdseye view conversion, video generation

# Advanced Lane-Lines project for Self driving car Nanodegree
'''PIPELINE:
            1.Camera calibration
            2.Undistort images
            3.Thresholding
            4.convert to 2D
            5.sliding point tracking and curvature
            6.Conversion back to regular image
            
    After this, apply pipeline to video'''


import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
import pickle
from camera_cal import calibrate_camera
from helpers import birdseye,hist,plot_all
from threshold import gray_blur,sobel_calc,sobel_thresh,mag_thresh,grad_thresh,color_thresh

# calibrate_camera() #run this command initially if never run before
with open('camera_cal.pickle','rb') as input_file:
    mtx,dist = pickle.load(input_file)

# undistorting test images
test_images = [] # list of test images
undst = [] # list of undistorted images
img_birds = [] # list of birdseye view images
# apply pipeline to test_images
for image in glob.glob('./test_images/*.jpg'):
    image1 = mpimg.imread(image) # read in image
    test_images.append(image1)
    dst = cv2.undistort(image1, mtx, dist, None, mtx) # undistort image
    undst.append(dst) # appended list of undistorted images
    img_color = color_thresh(dst,(140,255)) # color threshold
    img_mag = mag_thresh(dst,(100,200)) # magnitude threshold
    img_grad = grad_thresh(dst,(0.7,1.1)) # gradient threshold
    img_comb = np.zeros_like(img_grad)
    img_comb[((img_color == 1) | (img_mag == 1)) & (img_grad == 1)] = 1 # combined threshold
    img_per = birdseye(img_comb) # birdseye view of the detected lines
    img_birds.append(img_per) # appended birdseye list

# test image
image = img_birds[4]
# histogram for identifying peaks
histo = hist(image)
# image copy to plot lines on
img_copy = np.dstack((image,image,image))
# setup for finding and drawing
# divide the histogram into left and right sides for drawing two separate lines
midpoint = np.int(histo.shape[0]//2)
leftx_base = np.argmax(histo[:midpoint])
rightx_base = np.argmax(histo[midpoint:]) + midpoint # +midpoint added to provide offset

# sliding windows
nwindows = 9
margin = 100
minpix = 50

window_height = np.int(image.shape[0]//nwindows) # height of windows
# finding coordinates of nonzero pixels in the image
nonzeros = image.nonzero()
nonzerox = np.array(nonzeros[0])
nonzeroy = np.array(nonzeros[1])
# initializing start positions for the sliding window method
leftx_current = leftx_base
rightx_current = rightx_base
left_lane_inds = [] # for storing pixel indices to draw lines
right_lane_inds = []

for window in  range(nwindows):
    win_y_low = image.shape[0] - (window+1)*window_height
    win_y_high = image.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    #drawing the rectangle windows
    cv2.rectangle(img_copy,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0),thickness=4)
    cv2.rectangle(img_copy,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),thickness=4)

    # finding nonzero pixels within the windows
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)

    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    right_lane_inds.append(good_right_inds)

    # finding the mean x values for both lines in current window
    # and updating the x starting values for next window
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    print(leftx_current,rightx_current)
plt.imshow(img_copy)
plt.show()
    
    


# plot_all(undst)
# f,axs = plt.subplots(len(test_images), 2, figsize=(20,30))
# f.subplots_adjust(hspace = .2, wspace = .05)
# axs.ravel()

# for i in range(len(test_images)):
#     axs[i,0].imshow(test_images[i])
#     axs[i,1].imshow(img_birds[i])

# plt.tight_layout()
# plt.show()