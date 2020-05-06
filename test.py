## testing script for developing pipeline for single image

import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
import pickle
from camera_cal import calibrate_camera
from helpers import getTransformMatrices,birdseye,plot_all
from curvature import hist,sliding_window,get_polylines,cal_curvature
from draw_on_image import drawLine
from threshold import combined_thresh
from line_class import Line

def process_image(image1):
    '''Process the raw image and return an image with lane tracked, curvature and off center info'''
    global mtx,dist,M,Minv

    # image1 = mpimg.imread(image) # read in image
    image2 = cv2.undistort(image1, mtx, dist, None, mtx) # undistort image
    binary_image = combined_thresh(image2)
    new_img = np.copy(image2) # original undistorted image copied for drawing lanes 
    zero_warp = np.uint8(np.zeros_like(binary_image))
    color_warp = np.dstack((zero_warp,zero_warp,zero_warp))

    y_max = image2.shape[0] # max y value
    ploty = np.linspace(0,y_max-1,y_max)

    # get the fit lines
    leftx,lefty,rightx,righty = sliding_window(new_img)
    if len(leftx) != 0:
        left_line = np.polyfit(lefty,leftx,2)
    if len(rightx) != 0:
        right_line = np.polyfit(righty,rightx,2)
    # calculate curvature
    left_curvature = cal_curvature(leftx,lefty,ploty)
    right_curvature = cal_curvature(rightx,righty,ploty)

    # curvature sanity check
    if ((left_curvature > 5000) | (left_curvature < 100)):
        left_line = left_lane.left_fit
        left_curvature = left_lane.left_curvature
    else:
        left_lane.left_fit = left_line
        left_lane.left_curvature = left_curvature

    if ((right_curvature > 5000) | (right_curvature < 100)):
        right_line = right_lane.right_fit
        right_curvature = right_lane.right_curvature
    else:
        right_lane.right_fit = right_line
        right_lane.right_curvature = right_curvature
    try:
        left_fit = left_line[0]*ploty**2 + left_line[1]*ploty + left_line[2]
        right_fit = right_line[0]*ploty**2 + right_line[1]*ploty + right_line[2]
    except TypeError:
        left_fit = ploty**2 + ploty*2
        right_fit = ploty**2 + ploty*2

    # note arrays in a pts stack for use in cv2.fillPoly() function
    pts_left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # fill the polygon
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # draw edge lines of the polyfit lines
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=5)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,0,0), thickness=5)
    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (image2.shape[1], image2.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)

    # calculate vehicle position from center
    x_max = new_img.shape[1]
    expected_center = x_max/2
    veh_center = (left_fit[-1] + right_fit[-1])/2
    off_center = (veh_center - expected_center) * 3.7/660
    if off_center < 0:
        message = 'Vehicle is ' + '{:+.2f} m'.format(off_center) + ' left of center'
    else:
        message = 'Vehicle is ' + '{:+.2f} m'.format(off_center) + ' right of center'

    # adding required text onto the output images
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'Left line curvature: {:.0f} m'.format(left_curvature)
    cv2.putText(result,text1,(50,50),font,1,[255,255,255],2)
    text2 = 'Right line curvature: ' + '{:.0f}'.format(right_curvature) + 'm'
    cv2.putText(result,text2,(50,100),font,1,[255,255,255],2)
    cv2.putText(result,message,(50,150),font,1,[255,255,255],2)

    return result

# undistorting test images
test_images = [] # list of test images
img_birds = [] # list of birdseye view images
# apply pipeline to test_images
for image in glob.glob('./test_images/*.jpg'):
    image1 = mpimg.imread(image) # read in image
    test_images.append(image1)

# initial variables needed
with open('camera_cal.pickle','rb') as input_file:
    mtx,dist = pickle.load(input_file)
M = getTransformMatrices()
Minv = np.linalg.inv(M)
# initializing lines
left_lane = Line()
right_lane = Line()
image = test_images[5]

plt.imshow(process_image(image))
plt.show()

