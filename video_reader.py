''' extracting frames from video to improve advanced lane lines project'''
import os
import cv2
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# reading frames from video
video = cv2.VideoCapture('challenge_video.mp4')

# check if path exists. if not, create a path to store images in
if not os.path.exists('video_2_image'):
    os.makedirs('video_2_image')

images = [] # empty list to store the frames
index = 0
while(True):
    ret,frame = video.read()
    if not ret:
        break

    store = './video_2_image/frame' + str(index) + '.jpg'
    print('doing something with' + store)
    index += 1