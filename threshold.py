import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def gray_blur(image):
    '''Takes a undistorted image and returns a gray Gaussian blurred image'''
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT) # blur image
    return blurred

def sobel_calc(image,kernel_size=21):
    '''return the sobelx and sobely values'''
    blur = gray_blur(image)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=kernel_size)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=kernel_size)
    return sobelx,sobely

def sobel_thresh(image,axis,thresh=(20,100)):
    '''Takes in an blurred gray image and gives out a binary thresholded image'''
    blur = gray_blur(image)
    sobelx,sobely = sobel_calc(blur,21)
    binary_output = np.uint8(np.zeros_like(blur)//255)
    # Sobel operations
    threshold = [thresh[0],thresh[1]]
    if axis == 'x':
        abso_x = np.absolute(sobelx)
        scaledx = np.uint8(255*abso_x/np.max(abso_x))
        binary_output[(scaledx >= threshold[0]) & (scaledx <= threshold[1])] = 1
    elif axis == 'y':
        abso_y = np.absolute(sobely)
        scaledy = np.uint8(255*abso_y/np.max(abso_y))
        binary_output[(scaledy >= threshold[0]) & (scaledy <= threshold[1])] = 1
    return binary_output

def mag_thresh(image,thresh=(20,100)):
    '''Takes in an blurred gray image and returns a magnitude thresholded binary output'''
    sobelx,sobely = sobel_calc(image,21)
    blur = gray_blur(image)
    binary_output = np.uint8(np.zeros_like(blur)//255)
    threshold = [thresh[0],thresh[1]]
    abso = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*abso/np.max(abso))
    binary_output[(scaled > threshold[0]) & (scaled <= threshold[1])] = 1
    return binary_output

def grad_thresh(image,thresh=(0,np.pi/2)):
    '''take in a blurred gray image and return the gradient binary output image
    based on thresholds specified'''
    sobelx,sobely = sobel_calc(image,21)
    abso_x = np.absolute(sobelx)
    abso_y = np.absolute(sobely)
    grad_sobel = np.arctan2(abso_y,abso_x)
    binary_output = np.uint8(np.zeros_like(grad_sobel)//255)
    binary_output[(grad_sobel >= thresh[0]) & (grad_sobel <= thresh[1])] = 1
    return binary_output

def color_thresh(image,thresh=(60,120)):
    '''take an undistorted image and give a color thresholded image'''

    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    s = hls[:,:,2] # s channel provides consistent results
    copy = np.copy(s)
    copy[(s > thresh[0]) & (s <= thresh[1])] = 1
    return copy

def combined_thresh(image):
    '''apply decided thresholds on the undistorted image to get binary output'''
    img_color = color_thresh(image,(160,255)) # color threshold
    img_mag = mag_thresh(image,(50,255)) # magnitude threshold
    img_grad = grad_thresh(image,(0.38,1.15)) # gradient threshold
    img_comb = np.zeros_like(img_grad)
    img_comb[((img_color == 1) | (img_mag == 1)) & (img_grad == 1)] = 1 # combined threshold
    return img_comb