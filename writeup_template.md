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
[image0]: ./camera_cal/test_image.jpg "distorted"
[image1]: ./output_images/chessboard_undistorted.png "Undistorted_board"
[image2]: ./output_images/road_undistorted.png "Undistorted_road"
[image3]: ./output_images/combined_binary.png "binary threshhold"
[image4]: ./output_images/image_rectangale.png "image_rectangale"
[image5]: ./output_images/warped_image.png "warped_image"
[image6]: ./output_images/binary_warped_line.jpg "Road yellow line"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


---
### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients with an example of a distortion corrected calibration image.

The code for this step is called `Camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I also save these two matrixes using `np.savez` such that I can use them later. Then, I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
![alt text][image0] |  ![alt text][image1]
<p align="center">
  <img width="460" height="300" src="./camera_cal/test_image.jpg">
</p>

<p align="center">
  <img width="460" height="300" src="[image0]">
</p>

### Pipeline (single images)

The code for this step is called `Line_detection_advanced.py`.  
Initialy, it loads `mtx`, and `dist` matrices from camera calibration step.

#### 1. Apply a distortion correction to raw images.

Uisng the saved mtx, dist from calibration, I have undistorted an image from a road:
![alt text][image2]

#### 2. Create a thresholded binary image using color transforms and gradients.

(TODO: Which line of code?)

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 30 through 53 in `Line_detection_advanced.py`). 

For gradient thresholds, the code includes a function called `grad_thresh`. First, I converted the image into grayscale `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`. Note that if you are using `cv2.imread`, you should use `cv2.COLOR_BGR2GRAY`. But is you are using `matplotlib.image.imread`, you shoud use `cv2.COLOR_BGR2GRAY`. Then, I took the derivative in x direction, using `cv2.Sobel` (Why? Because vertical lines can be detected better using gradient in the horizontal direction). Then, I scaled its magnitude into 8bit `255*np.absolute(sobelx)/np.max(abs_sobelx)`, and conervetd to `np.unit8`. At the end, to generate the binary mage, I used `np.zeros_like`, and applied the threshhold.

For color threshhold, the code includes a function called `color_thresh`. I used HLS colorspace using `cv2.cvtColor(img, cv2.COLOR_BGR2HLS)`. (Why? because yellow and white colors can be detected well in S space). Then, I created the binary image `np.zeros_like`, and applied the threshhold on S channel.

At the end, I have combined the two binary threshholds, and here is an example of my output for this step.

![alt text][image3]

#### 3. Perform a perspective transform.

The code for my perspective transform includes a function called `warp()`, which appears in lines 55 through 68 in the file `Line_detection_advanced.py` (output_images/Line_detection_advanced.py). The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

#### 4. Identify lane-line pixels and fit their positions with a polynomial
To find lane pixels, a function called `find_lane_pixels()` is defined. First the histogram of the bottom half of the image along the vertical axis is computed usin `npsum`. 
Then the peaks in the left half side and right half side of the historam is computed as the initial estimate of the left and right lines resectively.
Then, the number of sliding windows `nwindows` and horizontal margin `margin` and the minimum nmber of pixels `minpix`.

Then I defined  `for` loop.
I start processing the bottom windows. 
The goal is to find the horizontal position of center of left and right windows which is used to found the boundries of the next window inside the for loop. As well as recognizing left and right lines oixel poistion.
First, the vertices of each left and right windows are computed. Then left and rectangles is plotted on the image uisng `cv2.rectangle`, and specefying two opposite vertices of a rectangle.
Then the indices of nonzero pixels in x and y directions within the windows are determined. 
I append them to the left and right lists of indices, uisng `np.append`. 
If the minimum number of recognized indices in left and right lists are more than `minpix`, I update the position of center of the left and right windows.
I continue to process the next windows which is the window above the bottom window. I continue till raach the `nwindows` which covers all the image along y axis. (TODO: really all th eimage?)

after the loop ends, I concatenate the arrays of indices (previously was a list of lists of pixels), using `np.concatenate`. Finally, I extract the left and right line pixel positions as the output of `find_lane_pixels`.

The next step is to fit a 2nd order polynomial uisng `np.polyfit` to the output of the prevoius function `find_lane_pixels`. 
To do this, I defined a function called `fit_polynomial()`.
To draw polynomials on the image, first I generate x and y values for plotting, using `np.linspace`. then use `fit[0]*ploty**2 + fit[1]*ploty + fit[2]` to have all points on the line for left and right lines. To plot them on the image, I use `plt.plot`. Also, I visualize the whole left and right windowes on the images.

The output of the last function is the fllowing figure:
(TODO: for some reason the yellow ones have not been saved)

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
