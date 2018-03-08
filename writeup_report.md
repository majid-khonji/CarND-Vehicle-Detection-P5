## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/slide.png
[image4]: ./output_images/img_pipeline.png
[image5]: ./output_images/heat.png
[image6]: ./output_images/labels.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained after the section entitled "Experimentation with different Parameters" of the IPython notebook `car_detect.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images in the first cell after the section "Experimentation with different Parameters". I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image2]

Here is an example using the `YCrCb` color space and HOG parameters of :

```colorspace = 'YUV'
orient = 12
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
 
```


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and selected the best based on SVM classification test accuracy.  I noticed more distinctive HOG when `pix_per_cell` is increased to 16. The color space `YCrCb` performs reasonably well for classification accuracy (accuracy numbers are commented in the cell below section "Experimentation with different Parameters."  Several combinations of HOG parameters (as well as other parameters) result accuracy ranging from 95% - 98%, which is reasonable for our application (shown in the cell below section "Training Classifier". 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training code is located after section "Training Classifier."

I trained a linear SVM using the following parameters for the feature  extraction:

```orient = 12
color_space = 'YCrCb'
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'

spatial_size = (16, 16)  
hist_bins = 16    
spatial_feat = True  
hist_feat = True  
hog_feat = True  
y_start_stop = [400, 700]  

```

After normalizing the data,  the classifier gave an accuracy of 98.2%. However, later when augmented my training data with false positive patches, the accuracy dropped to 97.8%, which is still sufficiently good. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is located after section "Search & Predict". I consider a slight modification of  a lesson functions `find_car` as a subroutine in a new function called `multi_level_windows_fast` (located in the second cell after the section). The function invokes 4 levels of  `find_car` :

![alt text][image3]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

I used the following parameters for the four levels, where cell_steps is `cells_per_step` parameter:

```
    scales = [1,1,1.4,3]
    cell_steps = [1,1,1,2]
    y_starts = [400,400, 400, 400]
    y_stops =  [480,500, 600, 700]
```

The values are set empirically in parallel with different heat map thresholds. 

 Here are some example images:

![alt text][image4]
---

To reduce false positives, I manually added wrongly classified patches (see the cells below False Positive section). There are some false positives in the images above; however, through thresholding (color, shape, and area) techniques, I managed to obtain good result for the video.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code is provided under section "Heatmap & Thresholding." I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

For video , in addition to heatmap thresholding, I used the following tricks (the video pipeline is located under section "Final Video Pipeline":

* Instead of calculating heatmap for each frame, I sum the heatmap of a window of time (`heat_list` of size `queue_length`), and then threshold on that
* Skip every `skip_every` frames: to speed up computation and to mitigate false positives 
* Area thresholding: for boxes in the lower part of the image, the area should be sufficiently large. See `area_threshold` located  in the cell below section Heatmap
* Shape thresholding: the ratio between the height and the width of a box shouldn't be very high. See `shape_threshold`  located  in the cell below section Heatmap

Here's an example result showing the heatmap from a series of test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from test1.jpg:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
Shown in the above figure with heatmaps



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First of all each training image for cars I used contain a complete picture of a car. Even though I get high classification accuracy (based on 20% test split), that number doesn't reflect the actual performance. It would be much better if I use a data set that include portions of vehicle images, not only the complete picture from the behind, which would work better with the sliding window technique.

To pursue the project further, I think it would be useful to devise a method for vehicle tracking. Vehicles should be detected in the lower and higher portion of a frame, while in the middle, we only track those detected. I think we can use a histogram technique, similar to lane detection, to detect true positive images over a time window. For example, consider the aggregation of  bboxes of 10 frames (time window of length 10), then we calculate a histogram on a lower and an upper region for the frame. If we get a high peak (exceeding a threshold), then a new vehicle is detected at that region. For the area in between, we search for a vehicle only in the nearby area of that in the previous frame.

