import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def warp(img, src, dst, img_size):

    #compute the perspective trasform, matrix M
    M = cv2.getPerspectiveTransform(src, dst)

    # creat warped image -uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_LINEAR)

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

def fit_polynomial(out_img, leftx, lefty, rightx, righty, size_binary_warped_0):

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, size_binary_warped_0-1, size_binary_warped_0 )
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
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    color=[255, 0, 0]
    thickness=2
    cv2.line(out_img, (left_fitx[0].astype(int), ploty[0].astype(int)),\
     (left_fitx[-1].astype(int), ploty[-1].astype(int)), color, thickness)
    cv2.line(out_img, (right_fitx[0].astype(int), ploty[0].astype(int)),\
     (right_fitx[-1].astype(int), ploty[-1].astype(int)), color, thickness)

    return out_img, ploty, left_fitx, right_fitx

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def unwarp(img, src, dst, img_size):

    # Could compute the inverse by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # creat warped image -uses linear interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size,flags=cv2.INTER_LINEAR)

    return unwarped

def Lane_Finding_Pipeline_Image_Advanced(image_road):

    data = np.load('calib_info.npz')
    mtx = data['mtx']
    dist = data['dist']
    #TODO: write the matrices here

    # undist_roadorting the test image_road:
    undist_road = cv2.undistort(image_road, mtx, dist, None, mtx)

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(image_road)
    # ax1.set_title('Original Image', fontsize=10)
    # ax2.imshow(undist_road)
    # ax2.set_title('Undistorted Image', fontsize=10)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # mpimg.imsave("road_undistorted.png", undist_road)
    # plt.show()

    # Note: img is the undistorted image
    img = np.copy(undist_road)

    sx_binary = grad_thresh(img, thresh=(20,100))
    s_binary = color_thresh(img, thresh=(170,255))
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1

    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(sx_binary)
    # ax1.set_title('grad thresh binary (sobel x)', fontsize=10)
    # ax2.imshow(s_binary)
    # ax2.set_title('color thresh binary (S from HSV)', fontsize=10)
    # ax3.imshow(combined_binary)
    # ax3.set_title('grad & color combined', fontsize=10)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # mpimg.imsave("thresh_cmbined_binary.png", combined_binary)
    # plt.show()

    # Define calibration box in source (original) and destination
    # (desired, warped coordinates)
    img_size = (img.shape[1], img.shape[0])

    # 4 source image points
    src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],#top left
    [((img_size[0] / 6) - 10), img_size[1]],#bottomleft
    [(img_size[0] * 5 / 6) + 45, img_size[1]],# bottom right
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])# top right

    # 4 desired coordinates
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    # get perspective transform of the binary image
    binary_warped = warp(combined_binary, src, dst, img_size)
    # plt.imshow(binary_warped)
    # plt.title('binary warped', fontsize=10)
    # mpimg.imsave("binary_warped.png", binary_warped)
    # plt.show()

    # For fun: get perspective transform of the original road image
    # plt.plot(src[0,0],src[0,1],'.')
    # plt.plot(src[1,0],src[1,1],'.')
    # plt.plot(src[2,0],src[2,1],'.')
    #plt.plot(src[3,0],src[3,1],'.')
    #plt.plot(dst[0,0],dst[0,1],'.')
    #plt.plot(dst[1,0],dst[1,1],'.')
    #plt.plot(dst[2,0],dst[2,1],'.')
    #plt.plot(dst[3,0],dst[3,1],'.')
    # road_rectangale=cv2.line(undist_road, (src[0,0],src[0,1]), (src[1,0],src[1,1]), (0, 255, 0) , 2)
    # road_rectangale=cv2.line(road_rectangale, (src[3,0],src[3,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
    # road_rectangale=cv2.line(road_rectangale, (src[0,0],src[0,1]), (src[3,0],src[3,1]), (0, 255, 0) , 2)
    # road_rectangale=cv2.line(road_rectangale, (src[1,0],src[1,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
    # plt.imshow(road_rectangale)
    # plt.title('road with rectangle', fontsize=10)
    # mpimg.imsave("image_rectangale.png", road_rectangale)
    # plt.show()

    # warped_road = warp(road_rectangale, src, dst, img_size)
    # plt.imshow(warped_road)
    # plt.title('warped road with rectangle', fontsize=10)
    # mpimg.imsave("warped_image_rectangale.png", warped_road)
    # plt.show()

    # Find our lane pixels first
    leftx, lefty, rightx, righty, binary_warped_pixel = find_lane_pixels(binary_warped)
    # plt.imshow(binary_warped_pixel)
    # plt.title('binary warped pixel', fontsize=10)
    # mpimg.imsave("binary_warped_pixel.png", binary_warped_pixel)
    # plt.show()

    # Fit a polynomial
    binary_warped_line, ploty, left_fit_cr, right_fit_cr = fit_polynomial(binary_warped_pixel, leftx, lefty, rightx, righty, binary_warped.shape[0])
    # plt.imshow(binary_warped_line)
    # plt.title('binary warped line', fontsize=10)
    # plt.show()
    #mpimg.imsave("binary_warped_line.png", binary_warped_line)

    # print(np.shape(left_fit_cr))#, right_fit_cr
    # print(left_fit_cr[0])
    # print(left_fit_cr[-1])

    uuwarped_binary = unwarp(binary_warped_line, src, dst, img_size)
    # plt.imshow(uuwarped_binary)
    # plt.title('unwarped binary', fontsize=10)
    # mpimg.imsave("uuwarped_binary.png", uuwarped_binary)
    # plt.show()

    # Fit a polynomial
    # warped_road_line, ploty, left_fit_cr, right_fit_cr = fit_polynomial(warped_road, leftx, lefty, rightx, righty, binary_warped.shape[0])
    # plt.imshow(warped_road_line)
    # plt.title('warped road line 1', fontsize=10)
    # plt.show()
    # mpimg.imsave("warped_road_line1.png", warped_road_line)
    # uuwarped_road1 = unwarp(warped_road_line, src, dst, img_size)
    # plt.imshow(uuwarped_road1)
    # plt.show()
    # mpimg.imsave("uuwarped_road1.png", uuwarped_road1)

    # plot lines on the warped road image:
    # warped_road_0 = np.zeros_like(warped_road)
    # fig1 = plt.gcf()
    # plt.imshow(warped_road_0)
    # plt.plot(left_fit_cr, ploty, color='yellow')
    # plt.plot(right_fit_cr, ploty, color='yellow')
    # plt.title('warped road line', fontsize=10)
    # fig1.savefig('warped_road_line.png', dpi=100)
    # mpimg.imsave("warped_road_line.png", warped_road_0)
    # plt.show()
    #
    # uuwarped_road = unwarp(warped_road_0, src, dst, img_size)
    # plt.imshow(uuwarped_road)
    # plt.title('unwarped road line', fontsize=10)
    # mpimg.imsave("uuwarped_road2.png", uuwarped_road)
    # plt.show()

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # calculate the curve raduis in meters
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix)
    #print(left_curverad, 'm', right_curverad, 'm')
    #calculate average of curvature raduis
    R_curve = (left_curverad + right_curverad)/2
    text_R = '{} meters raduis of curvature'.format(round(R_curve,2))

    lane_mid = (left_fit_cr[0] + right_fit_cr[0])/2
    middle_img = img_size[0]/2
    car_off_center = (lane_mid - middle_img)*xm_per_pix
    #print(car_off_center)

    if car_off_center >= 0:
        text_C = '{} meters left of center'.format(round(car_off_center,2))
    else:
        text_C = '{} meters right of center'.format(round(-car_off_center,2))

    # # font
    # font = cv2.FONT_HERSHEY_SIMPLEX
    #
    # # org
    # org = (50, 50)
    #
    # # fontScale
    # fontScale = 1
    #
    # # Blue color in BGR
    # color = (255, 0, 0)
    #
    # # Line thickness of 2 px
    # thickness = 2

    # Using cv2.putText() method
    cv2.putText(undist_road, text_C, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(undist_road, text_R, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 0, 0), 2, cv2.LINE_AA)
    # Create a "color" binary image to combine with line image
    result = cv2.addWeighted(undist_road, 0.5, uuwarped_binary, 1., 0.)

    return result

# Read in the distorted test image
image_path = '../test_images/straight_lines1.jpg'
image = cv2.imread(image_path)

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

result = Lane_Finding_Pipeline_Image_Advanced(image)
plt.imshow(result)
plt.show()
mpimg.imsave("result.png", result)


"""
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

video_out = 'project_video_output.mp4'
# white_output = 'test_videos_output/challenge2.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip1 = VideoFileClip("../project_video.mp4").subclip(0,2)
#clip1 = VideoFileClip("test_videos/challenge.mp4")
clip = clip1.fl_image(Lane_Finding_Pipeline_Image_Advanced) #NOTE: this function expects color images!!
clip.write_videofile(video_out, audio=False)
"""
