import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
import pickle

def calibrate_camera():
    '''use calibration images to automatically calculate the undistortion coefficients'''
    # initialzing
    imgpoints = [] # 2d image world
    objpoints = [] # 3d image world
    objp = np.zeros((6*9,3), np.float32) # size of array is 9 by 6
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x and y coordinates
    cal_set = []
    for image in glob.glob('./camera_cal/calibration*.jpg'): # folder path
        n = mpimg.imread(image)
        cal_set.append(n)
        img = np.copy(n)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret,corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            img = cv2.drawChessboardCorners(img,(9,6),corners,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (6,9) , None, None)
    
    
    with open(b'camera_cal.pickle','wb') as output_file:
        pickle.dump([mtx,dist],output_file)