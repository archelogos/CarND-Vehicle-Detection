# Vehicle Detection

In this project our main goal was to design, create and build a able pipeline to detect and track vehicles in a given video from a
camera mounted in the front of a car.

In order to achieve this goal, we applied some computer vision and machine learning techniques. The first important phase was to define and extract the relevant features from the images applying gradients and transforms to them.

Once the features were properly extracted we trained a Linear SVM classifier with the GTI vehicle image database and the KITTI vision benchmark suite datasets.

Finally, applying a sliding-window technique, we searched for vehicles in the images, pointed them out drawing a surrounding and clear rectangle.

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./others/car_not_car.png
[image2]: ./output_images/how_visualization.jpg
[image3]: ./output_images/windows.jpg
[image4]: ./output_images/vehicle_detection.jpg
[image5]: ./output_images/heatmaps.png
[image7]: ./others/output_bboxes.png
[video1]: ./result.mp4

## Histogram of Oriented Gradients (HOG)

### HOG features extraction

First, we needed to explore the datasets to understand the problem we were supposed to solve. Both datasets contain a bunch of examples of not-car and car images. The images are in color and have a size of 64 by 64 pixels. The images can only belong to one of two classes, it's a car image or is not. The **labels** are given by the name of the folder where the images are stored.

![alt text][image1]

As it was mentioned before, we wanted to extract the relevant features from the images in order to be able to detect vehicles in the images. Because the HOG technique works very well in this scenarios, it was decided to apply it to the images.

However, the key point here was not to decide if apply HOG or not but how it was applied, or in other words select a set of proper values.

It also important to mentioned that the parameters not only have an effect in terms of detection but also in terms of performance. It's not the same select 9 than 11 orientations. As well as it isn't the same either select 32 pixels_per_cell than 8.
The feature extraction parameters selection is explained in the following section.

To introduce some results, in the following image it can be seen an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

```python
cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

# Check the HOG extraction

cars = glob.glob('./data/vehicles/*/*.png')
notcars = glob.glob('./data/non-vehicles/*/*.png')

random_index = np.random.randint(0, len(cars))

img = cars[random_index]
features, hog_image = features_for_vis(mpimg.imread(img), cspace=cspace,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

fig = plt.figure()
plt.subplot(1, len(hog_image)+1, 1)
plt.imshow(mpimg.imread(img))
plt.title('Example Car Image')
for idx, hog_img in enumerate(hog_image):
  plt.subplot(1, len(hog_image)+1, idx+2)
  plt.imshow(hog_img, cmap='gray')
  plt.title('HOG Visualization')
```

![alt text][image2]

### Feature extraction Parameters

In the first part, some helper functions were defined to visualize the HOG features in images. But after that some functions were redefined
and final parameters were the following:

```python
cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

At the beginning we tried with HLS and HSV color spaces but after several trial and error phases it seemded that
YUV and YCrCb performed better. Finally, the YCrCb was selecetd with a number of 9 HOG orientations. It could be said that is
a balanced number between extracting enough number of features but no so many that could slow down the training phase.

As it can be seen, it was decided to select all of the three HOG channels corresponding with each color channel. This improved a lot the model performance

The spatial and color extraction were also taken into account to fit the final features.

The final feature vector length was of 8096 in total.

When the feature vectors were created, we did some relevant steps:

- Fit a per-scaler and normalize the features from the transformations and HOG extraction
- Split up data into randomized training and test sets thanks to the ```train_test_split()``` method

### Training phase

Our model is based on a Linear SVM Classifier.

The training phase was successfully passed with a 99.25% of accuracy in the test set in less than 60 seconds. As it known, the SVM performs very well in image classfication problems so we didn't need to tried several different models.

As it can be seen, it weren't tweak the default params of the SVM (alpha or C)

```python
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```

Once the model was properly trained, the model values were stored in a .pkl file to avoid repeating the same steps over and over and continue with the project from there.

## Sliding Window Search

### Windows Set up

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

### Pipeling on images

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


## Video Implementation

### Video pipeline
Here's a [link to my video result](./project_video.mp4)


### Vehicle tracking and False positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


---


## Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
