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

# get perspective transform
warped_im = warp(combined_binary, src, dst)


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
# plt.imshow(image_rectangale)
# plt.show()
mpimg.imsave("image_rectangale.png", image_rectangale)
# get perspective transform from the original image
warped_image = warp(image_rectangale, src, dst)
# plt.imshow(warped_image)
# plt.show()
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
"""

#visualize undistortion
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(combined_binary)
ax2.set_title('warped image')
ax2.imshow(warped_im)
plt.show()
mpimg.imsave("warped.png", warped_im)

#visualize undistortion
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(image_rectangale)
ax2.set_title('warped image')
ax2.imshow(warped_image)
plt.show()
mpimg.imsave("warped_road.png", warped_image)
