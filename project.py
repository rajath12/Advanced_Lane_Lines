# TO-DO: find curvature value, anti-birdseye view conversion, video generation

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
import cv2
import numpy as np
import pickle
from threshold import combined_thresh
from curvature import cal_curvature,sliding_window
from draw_on_image import drawLine
from helpers import plot_all,getTransformMatrices
from moviepy.editor import VideoFileClip
from line_class import Line

def process_image(image1):
    '''draw on original image'''
    with open('camera_cal.pickle','rb') as input_file:
        mtx,dist = pickle.load(input_file)

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
    # get Minv matrix for converting perspective transformed image
    M,Minv = getTransformMatrices()
    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (image2.shape[1], image2.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)

    # calculate vehicle position from center
    x_max = new_img.shape[1] * 3.7/200
    expected_center = x_max/2
    y_max = y_max * 30/720
    # calculate image value at the bottom of the image
    leftvalue = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    rightvalue = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    veh_center = leftvalue + ((rightvalue - leftvalue)/2)
    off_center = veh_center - expected_center
    print(x_max,y_max,expected_center,off_center)

    # adding required text onto the output images
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'Left line curvature: {:.0f} m'.format(left_curvature)
    cv2.putText(result,text1,(50,50),font,1,[255,255,255],2)
    text2 = 'Right line curvature: ' + '{:.0f}'.format(right_curvature) + 'm'
    cv2.putText(result,text2,(50,100),font,1,[255,255,255],2)
    text3 = 'Off from center: ' + '{:.0f}'.format(off_center) + 'm'
    cv2.putText(result,text3,(50,150),font,1,[255,255,255],2)

    return result

left_lane = Line()
right_lane = Line()
white_output = 'output_video/project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(0,2)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)