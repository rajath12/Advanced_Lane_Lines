import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from helpers import birdseye
from threshold import combined_thresh

def hist(image):
    '''take in a perspective transformed image and draw the histogram based on 
    lane lines found near the bottom'''
    bottom_half = image[(image.shape[0]-image.shape[0]//2):,:]
    hist = np.sum(bottom_half, axis=0)
    return hist

def sliding_window(img):
    '''Takes in an undistorted image and returns sliding windows and x,y coordinates for drawing the lines '''
    image = combined_thresh(img)
    img1 = birdseye(image)
    histo = hist(img1)
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
    nonzeros = img1.nonzero()
    nonzeroy = np.array(nonzeros[0])
    nonzerox = np.array(nonzeros[1])
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
        cv2.rectangle(img_copy,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),thickness=4)
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

    # merge all left and right window arrays into one
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    # extract x and y dimensions of the pixels to draw the two lines
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    return left_x,left_y,right_x,right_y

def draw_polylines(image):
    '''Take in perspective transformed binary image and draw polynomial lines along lanes'''
    leftx,lefty,rightx,righty = sliding_window(image)
    # here x is to be found and y is the variable
    left = np.polyfit(lefty,leftx,2)
    right = np.polyfit(righty,rightx,2)

    # let ploty be the variable
    # ploty = np.linspace(0,image.shape[0]-1,image.shape[0])
    # try:
    #     left_line = left[0]*ploty**2 + left[1]*ploty + left[2]
    #     right_line = right[0]*ploty**2 + right[1]*ploty + right[2]
    # except TypeError:
    #     print('Function failed to fit a line')
    #     left = ploty**2 + ploty
    #     right = ploty**2 + ploty

    # out[lefty,leftx] = [255,0,0]
    # out[righty,rightx] = [0,0,255]

    # plt.plot(left_line,ploty,color='yellow')
    # plt.plot(right_line,ploty,color='yellow')

    return left,right

def cal_curvature(x_vals,y_vals,ploty):

    x_m = 3.7/900
    y_m = 15/720
    y_vals = y_vals[::-1]
    fit_curve = np.polyfit(y_vals*y_m,x_vals*x_m,2)
    y_eval = np.max(ploty)*y_m
    # Calculation of R_curve (radius of curvature)
    curverad = ((1 + (2*fit_curve[0]*y_eval + fit_curve[1])**2)**1.5)/np.absolute(2*fit_curve[0])

    return curverad