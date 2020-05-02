# Advanced Lane-Lines project for Self driving car Nanodegree
""" """
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
import pickle

with open('camera_cal.pickle','rb') as input_file:
    objpoints,imgpoints = pickle.load(input_file)

test_images = []
for image in glob.glob('./test_images/*.jpg'):
    test_images.append(mpimg.imread(image))

image1 = test_images[6]
img_size = (image1.shape[1], image1.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size , None, None)
dst = cv2.undistort(image1, mtx, dist, None, mtx)

plt.figure(1)
plt.subplot(121)
plt.imshow(image1)
plt.subplot(122)
plt.imshow(dst)
plt.show()
