import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def birdseye(image):
    ''' Takes in an undistorted image and returns the unwarped image image.'''    
    img_size = (image.shape[1],image.shape[0]) # be wary of this variable during implementation

    # source points from undistorted image and destination points
    src = np.float32([[556,450],[685,450],[980,650],[295,650]]) # found coordinates using cv2 lines
    offset = 250 
    dest = np.float32([
                    [offset,0],
                    [image.shape[1]-offset,0],
                    [image.shape[1]-offset,image.shape[0]],
                    [offset,image.shape[0]]])
    M = cv2.getPerspectiveTransform(src,dest) # perspective matrix
    warped = cv2.warpPerspective(image,M,img_size) # birds-eye view
    # dst = cv2.line(dst,(980,650),(295,650),[255,0,0],thickness = 2) # use for debugging coordinates
    return warped
    
def hist(image):
    '''take in a perspective transformed image and draw the histogram based on 
    lane lines found near the bottom'''
    bottom_half = image[(image.shape[0]-image.shape[0]//2):,:]
    hist = np.sum(bottom_half, axis=0)
    return hist
    
def plot_all(folder):
    '''takes in an image folder, do the appropriate stuff and plot all output images'''
    f,axs = plt.subplots(1,len(folder), figsize=(50,50))
    f.subplots_adjust(hspace = .2, wspace = .05)
    axs.ravel()
    i = 0
    for image in folder:
        axs[i].imshow(image)
        axs[i].axis('off')
        i += 1
    plt.tight_layout()
    plt.show()