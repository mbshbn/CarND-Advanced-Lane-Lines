import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data = np.load('calib_info.npz')
mtx = data['mtx']
dist = data['dist']

# Undistord a road image:
# Read in the distorted test image
image = cv2.imread('../test_images/straight_lines1.jpg')
# Undistorting the test image:
undist = cv2.undistort(image, mtx, dist, None, mtx)
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
mpimg.imsave("road_undistorted.png", undist)
"""
# Note: img is the undistorted image
img = np.copy(undist)

def grad_thresh(img, thresh=(20,100)):
    # Gradient thresholds:
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def color_thresh(img, thresh=(170,255)):
    # Color thresholds:
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def warp(img, src, dst ):

    #compute the perspective trasform, matrix M
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse by swapping the input parameters
    # minv = cv2.getPerspectiveTransform(dst, src)

    # creat warped image -uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_LINEAR)
    plt.imshow(warped)
    plt.show()

    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(out_img, leftx, lefty, rightx, righty):

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

sx_binary = grad_thresh(img, thresh=(20,100))
s_binary = color_thresh(img, thresh=(170,255))
# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(sx_binary)
combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1

# Define calibration box in source (original) and destination
# (desired, warped coordinates)
img_size = (img.shape[1], img.shape[0])

# 4 source image points
src = np.float32(
[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],#top left
[((img_size[0] / 6) - 10), img_size[1]],#bottomleft
[(img_size[0] * 5 / 6) + 60, img_size[1]],# bottom right
[(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])# top right

# 4 desired coordinates
dst = np.float32(
[[(img_size[0] / 4), 0],
[(img_size[0] / 4), img_size[1]],
[(img_size[0] * 3 / 4), img_size[1]],
[(img_size[0] * 3 / 4), 0]])

# get perspective transform of the binary image
binary_warped = warp(combined_binary, src, dst)

"""
# For fun: get perspective transform of the original road image
#plt.imshow(warped_image)
# plt.plot(src[0,0],src[0,1],'.')
# plt.plot(src[1,0],src[1,1],'.')
# plt.plot(src[2,0],src[2,1],'.')
#plt.plot(src[3,0],src[3,1],'.')
#plt.plot(dst[0,0],dst[0,1],'.')
#plt.plot(dst[1,0],dst[1,1],'.')
#plt.plot(dst[2,0],dst[2,1],'.')
#plt.plot(dst[3,0],dst[3,1],'.')
#plt.show()
image_rectangale=cv2.line(image, (src[0,0],src[0,1]), (src[1,0],src[1,1]), (0, 255, 0) , 2)
image_rectangale=cv2.line(image_rectangale, (src[3,0],src[3,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
image_rectangale=cv2.line(image_rectangale, (src[0,0],src[0,1]), (src[3,0],src[3,1]), (0, 255, 0) , 2)
image_rectangale=cv2.line(image_rectangale, (src[1,0],src[1,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
mpimg.imsave("image_rectangale.png", image_rectangale)
warped_image = warp(image_rectangale, src, dst)
"""
# Find our lane pixels first
leftx, lefty, rightx, righty, binary_warped_pixel = find_lane_pixels(binary_warped)

# Fit a polynomial
binary_warped_line = fit_polynomial(binary_warped_pixel, leftx, lefty, rightx, righty)

# plt.imshow(binary_warped_pixel)
# plt.show()
# mpimg.imsave("binary_warped_pixel.png", binary_warped_pixel)

# plt.imshow(binary_warped_line)
# plt.show()
mpimg.imsave("binary_warped_line.png", binary_warped_line)


"""
# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.tight_layout()
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
mpimg.imsave("combined_binary.png", combined_binary)

#visualize perspective transform on binary image
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(combined_binary)
ax2.set_title('warped image')
ax2.imshow(warped_im)
plt.show()
mpimg.imsave("warped.png", binary_warped)

#visualize perspective transform on original road image
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(image_rectangale)
ax2.set_title('warped image')
ax2.imshow(warped_image)
plt.show()
mpimg.imsave("warped_road.png", warped_image)
"""
