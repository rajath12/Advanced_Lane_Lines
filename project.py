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
import glob
import cv2
import numpy as np
import pickle
from draw_on_image import drawLine
from helpers import plot_all
from moviepy.editor import VideoFileClip
# calibrate_camera() #run this command initially if camera_cal.pickle does not exist

# # undistorting test images
# test_images = [] # list of test images
# # apply pipeline to test_images
# for image in glob.glob('./test_images/*.jpg'):
#     image1 = drawLine(image)
#     test_images.append(image1)

# plot_all(test_images)

# def videoPipeline(inputVideo, outputVideo):
#     """
#     Process the `inputVideo` frame by frame to find the lane lines, draw curvarute and vehicle position information and
#     generate `outputVideo`
#     """
#     def processimage(image):
#         return drawLine(image)
#     myclip = VideoFileClip(inputVideo)
#     clip = myclip.fl_image(processimage)
#     clip.write_videofile(outputVideo, audio=False)
             
# # Project video
# videoPipeline('project_video.mp4', 'video_output/project_video.mp4')

def process_image(image):
    return drawLine(image)
white_output = 'output_video/project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(0,10)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)