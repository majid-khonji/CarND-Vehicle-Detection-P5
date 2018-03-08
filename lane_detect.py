import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import pickle
import matplotlib.image as mpimg


### Global business
with open('wide_dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    dist = dist_pickle["dist"]
    mtx = dist_pickle["mtx"]

straight_line_img =  cv2.imread('test_images/test1.jpg')
straight_line_img = cv2.cvtColor(straight_line_img, cv2.COLOR_BGR2RGB)

# Warp parameters
h,w = straight_line_img.shape[:2]
src = np.float32([(678,443),(605,443),(285,665),(1019,665)]) # in x,y order not matrix convention
dst = np.float32([(1019-100,0),(285,0),(285,665),(1019-100,665)])
# src = np.float32([(726,475),(558,475),(278,665),(1022,665)])
# dst = np.float32([(1022,0),(278,0),(278,665),(1022,665)])
lane_height = 665
lane_width = 1019 - 100 - 285
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

vehicle_center_unwarp = (w/2 - 285)/(1019-285)* (1019 - 285 - 100)+ 285
print('img center, camera unwarp center', w/2,vehicle_center_unwarp)


#######

def remove_distortion(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshold_binary(img,s_thresh=(100, 255), l_thresh=(180,255), sx_thresh=(10,150)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
#     l_channel = hls[:,:,1]
    l_channel = lab[:,:,0]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((s_binary,l_binary, sxbinary)).astype('uint8')  * 255
    
    binary = np.zeros_like(s_channel)
    binary[(sxbinary == 1) | (s_binary + l_binary == 2)] = 1
    return color_binary, binary

def prespective_transform(img):
    global src,dst,h,w,M,Minv
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped

# helper functions
def _find_window_centroids(image, window_width, window_height, margin, min_px_per_box=35):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    last_l_center = 0
    last_l_offset = 0
    
    last_r_center = 0
    last_r_offset = 0
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        if np.argmax(conv_signal[l_min_index:l_max_index]) >= min_px_per_box:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            last_l_offset = l_center - last_l_center
        else:
            l_center = last_l_center + last_l_offset
        
        last_l_center = l_center    
            
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        assert(r_min_index != r_max_index)
        if np.argmax(conv_signal[l_min_index:l_max_index]) >= min_px_per_box:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            last_r_offset = r_center - last_r_center
        else:
            r_center = last_r_center + last_r_offset
        last_r_center = r_center
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
    return window_centroids, np.concatenate((l_sum, r_sum))


def _window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output



# detect line pixels by the rolling window technique
def find_line_pixels(warped, window_width = 80, window_height = 80, margin = 100, verbos=False ):
    window_centroids, hist_data = _find_window_centroids(warped, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.copy(warped)
        r_points = np.copy(warped)
        boxes = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = _window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = _window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 1) & ((l_mask == 1) ) ] = 255
            r_points[(r_points == 1) & ((r_mask == 1) ) ] = 255
            boxes[((l_mask == 1)) | ((r_mask == 1))] = 255

        zero_channel = np.zeros_like(warped) # create a zero color channel
        boxes = np.array(cv2.merge((boxes,boxes,boxes)),np.uint8) # make window pixels green
        warpage= np.dstack((l_points, zero_channel, r_points)).astype('uint8') # making the original road pixels 3 color channels
        fancy_output = cv2.addWeighted(warpage, 1, boxes, 0.3, 0.0) # overlay the orignal road image with window results
    # If no window centers found, just display orginal road image
    else:
#         fancy_output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        print("error")
    
    if verbos:
        return fancy_output, hist_data
    else:
        l_y,l_x = np.where(l_points == 255)
        r_y,r_x = np.where(r_points == 255)
    return l_x,r_x, l_y, r_y

class Line():
    def __init__(self, queue_length=10):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # polynomial coefficients of the last n fits
        self.recent_fit = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #n
        self.queue_length=5
        self.ploty = np.linspace(0, h, num= h)# to cover same y-range as image

    # updates line data and decides wether the line is consistant through time time window    
    def update(self,x,y, force_append=False):
        fit = np.polyfit(y, x, 2)
        fitx = fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2]
        self.allx = x
        self.ally = y
        self.current_fit = fit

        if self.best_fit == None:
            self.best_fit = fit
            self.bestx = fitx
            self.detected = True
            # future update: assure that the first line is the best possible
#             print('first time', fit)

#         print('a', fit)
#         print('diff', self.diffs[0] > .002 or self.diffs[1] > 1 or self.diffs[2] > 120, 'best', self.best_fit)
        if self.current_fit != None:
            self.diffs = np.abs(self.best_fit - self.current_fit)

        # bad line or not detected: Time inconsistancy
        if self.diffs[0] > .001 or self.diffs[1] > 1 or self.diffs[2] > 120 or self.current_fit == None:
            self.detected = False
        if self.detected or force_append: # good line
            self.recent_fit.append(fit)
        
        # truncate data to queue_length
        if len(self.recent_fit) > self.queue_length:
            self.recent_fit = self.recent_fit[len(self.recent_fit)-self.queue_length:]
            self.recent_xfitted = self.recent_xfitted[len(self.recent_fit)-self.queue_length:]
                      
        # smooth curve over time
        if len(self.recent_fit)>1:
            self.best_fit = np.average(self.recent_fit, axis=0)
            self.bestx = self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]
            #self.recent_xfitted.append(self.bestx)

#             print('b',len(self.recent_fit),self.best_fit)
#             print('--')
        else:
            self.bestx = fitx
            self.best_fit = fit
        self.update_lane_data()
        
    # reverts to the previous self.best_fit
    def revert_to_previous_fit(self):
        if len(self.recent_fit) > 1:
            self.recent_fit = self.recent_fit[:-1]
            if len(self.recent_fit) > 1:
                self.best_fit = np.average(self.recent_fit, axis=0)
            else:
                self.best_fit = self.recent_fit
                
            self.bestx = self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]
            self.update_lane_data()
    # updates radius_of_curvature and line_base_pos based on self.bestx
    def update_lane_data(self):
        global lane_width, lane_height,h,w
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 35/lane_height # meters per pixel in y dimension
        xm_per_pix = 3.7/lane_width # meters per pixel in x dimension
        
        ploty = self.ploty
        fit = self.best_fit
        
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        vehicle_loc = w/2
        lane_loc = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
        self.line_base_pos = (lane_loc-vehicle_center_unwarp) * xm_per_pix
    # fast line pixel detection using self.best_fit    
    def find_line_pixels_by_best_fit(self, img, margin = 50):
        pixels = img.nonzero()
        y = np.array(pixels[0])
        x = np.array(pixels[1])
        fit = self.best_fit
        fit_x = fit[0]*(y**2) + fit[1]*y + fit[2]

        y = y[((x > (fit_x - margin)) & (x < (fit_x + margin)))] 
        x = x[((x > (fit_x - margin)) & (x < (fit_x + margin)))]

        return x,y

def draw_lane(img, left_line, right_line):
    global Minv,h,w,lane_widthlane_width
    ploty = np.linspace(0, h, num=h)# to cover same y-range as image
    left_fitx = left_line.bestx
    right_fitx = right_line.bestx
  
    # extract data
    left_curverad = left_line.radius_of_curvature
    right_curverad = right_line.radius_of_curvature
    curvature = (left_curverad+right_curverad)/2
    shift = (left_line.line_base_pos + right_line.line_base_pos)/2
    lane_width = right_line.bestx[-1] - left_line.bestx[-1]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=30)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,0,255), thickness=30)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    text1 = str('Radius of curvature: {:04.1f}'.format(curvature)+'m')
    cv2.putText(result,text1,(50,50), cv2.FONT_HERSHEY_DUPLEX, 1.6,(255,255,255),2,cv2.LINE_AA)

    direction = 'left' if shift >= 0 else 'right'
    text2 = str('Vehicle is {:04.2f}'.format(np.abs(shift))+'m '+direction+" of center")
    cv2.putText(result,text2,(50,100), cv2.FONT_HERSHEY_DUPLEX, 1.8,(255,255,255),2,cv2.LINE_AA)
    return result    

    
    
left_line = Line()
right_line = Line()

# returns the final annotated image
def video_pipeline(img):
    global left_line, right_line

    undst = remove_distortion(img)
    warped = prespective_transform(undst)
    _, binary = threshold_binary(warped)

    leftx,rightx,lefty,righty = None,None,None,None
    if not left_line.detected or not right_line.detected:
        leftx,rightx,lefty,righty = find_line_pixels(binary)
#         print('slow:',len(rightx),len(righty))
        right_line.update(rightx,righty,force_append=True)
        left_line.update(leftx,lefty, force_append=True)
        width = np.abs(left_line.line_base_pos)+np.abs(right_line.line_base_pos)
        if width > 3.9 or width < 3: # too large or two small, consider the previous fit
#             print('lane width: ', width )
            left_line.revert_to_previous_fit()
            right_line.revert_to_previous_fit()
            left_line.detected = True
            right_line.detected = True


    if left_line.detected:
        leftx,lefty = left_line.find_line_pixels_by_best_fit(binary)
        left_line.update(leftx,lefty)
#         print('left fast:',len(leftx),len(lefty))
    if right_line.detected:
        rightx,righty = right_line.find_line_pixels_by_best_fit(binary)
#         print('right fast:',len(rightx),len(righty))
        right_line.update(rightx,righty)
    
    # sanity check with respect to the two lines
    # distance between lines too large or too small
    # check curveture of the two 

    result = draw_lane(undst,left_line,right_line)
    return result


if __name__ == "__main__":
    from IPython.display import HTML
    from moviepy.editor import VideoFileClip


    video_output = 'project_video_output.mp4'
    video_input = VideoFileClip('project_video.mp4')#.subclip(20,32)#.subclip(10,12)
    #video_input.save_frame("start.jpg", t=12) # saves the frame at time 

    processed_video = video_input.fl_image(video_pipeline)
    processed_video.write_videofile(video_output, audio=False)

#    img = mpimg.imread("test_images/test2.jpg")
#    result = video_pipeline(img)
#    #result = video_pipeline(img)
#    plt.imshow(result)
#    plt.savefig('test.jpg')
