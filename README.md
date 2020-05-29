## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/board_calibration.png "Undistorted"
[image2]: ./output_images/road_undistorted.jpg "Road Transformed"
[image3]: ./output_images/combined_threshold.jpg "Binary Example"
[image4]: ./output_images/warped_road "Warp Example"
[image5]: ./output_images/drawlines_sliding_window.png "Fit Visual"
[image6]: ./output_images/final_image.jpg "Output"
[video1]: ./output_video/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file 'camera_cal.py' and the mtx,dist variable values for undistortion is saved as a pickle file 'camera_cal.pickle'. 
I initialized the 'objpoints' as a 3d grid of 9 by 6 as required. Used the cv2.findChessBoardCorners() function to iterate through the camera calobration images. Based on the ret values, 'imgpoints' and 'objpoints' are updated. These points are then used to calculate the calibration and distortion coefficients using cv2.calibrateCamera() function. These coefficients are then used in the cv2.undistort() function to get the undistorted images. A sample undistorted image is shown below.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The piclkled file is accessed to get the mtx and dist coefficients and cv2.undistort() function is used on the image to get the undistorted image. The undistortion is evident on closer inspection of the dashboard of the car in the image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I played with color, gradient, magnitude thresholding on the set of images and found a set of thresholding values that provided consistent results. The therholding functions are in the 'threshold.py' file. For each image, the combined threshold with set threshold values is called to obtain the binary output image as shown below.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This is contained in the 'helpers.py' file.
To get the transformation matrix M using the cv2.getPerspectiveTransform(src,dest), the src and dest points were decided by visual inspection of the lane lines on the undistorted image. To verify, lines were drawn to make sure the src points were on the lane lines. Appropriate dest points were then decided and the warped image was drawn using the cv2.warpPerspective() in the birdseye() function. The output of this process is shown below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Next step is to get process this perspective transformed image to draw the polynomial lines on the iamge. This is implemented in the 'curvature.py' file using the sliding_window() and get_polylines() functions.
Histogram of the perspective binary image is used to identify the location of concentration of pixels in the y direction within small windows moving from the bottom to the top of the image. These pixel values are stored and used to create the left and right polylines of the lane using the np.polyfit() function. The output of this process is shown:
 
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

By inspection of the birdseye view, the transformation of pixels to metres in the x-direction was found and by inspecting the final output image of the tracked lane, the value in the y-direction was found.
Using this information, the calc_curvature() function in the 'curvature.py' file calculates the curvature of the lines. The input to this funciton is x and y values obtained after applying sliding_window() function. Using this information, the radius of curvature is caluclated.

The assumption at the start of this project is that the camera is mounted at a central positon. So, the expected center of the lane is half of the image width. The vehicle center is the average of the left and right x values stored for the first iteration of the sliding_window() function. The vehicle is then said to be on right of the center if the off_center is positive and left of the center if the off_center value is negative.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the above mentioned steps for single images in the 'test.py' file. The images were loaded and the process_image() function containing the pipeline is called. The above procedure is applied on the image. Values with curvature > 5000 and < 100 are ignored as part of a sanity check. The output is displayed as shown with the tracked lane and info displayed:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The above process is repeated this time for the project_video in the file 'project.py' and the video output is stored in the location './output_video/project_video_output.mp4'.

![alt text][video]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Current drawbacks and suggested improvements:
1. Because the sliding window is calculated every single time, the processing speed is slow. This can be improved by applying a look ahead filter to only apply the sliding window on the part of the image that has changed from the previous one.
2. Tracking the broken lines sometimes creates bad fits due to the sliding window not having enough information about the previous fit. Comparing current fit to previous fits will help in this and also averaging over a few fits will provide stability to the lines. 
