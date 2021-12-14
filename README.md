# Lane_Lines_and_Car_Detection

## Faster R-CNN
![Faster RCNN](https://user-images.githubusercontent.com/82511782/146067210-19adab46-54dd-4e83-bb5b-de7daab5e700.png)

The architecture of Faster R-CNN consists of 2 modules : 

1. RPN, for Generating region proposals
2.Fast R-CNN for detecting objects in proposed regions.

The input images are HxWxDepth Tensors which are passed to CNN up until intermediate layer. This returns Convolutional Feature Map of the image.

Then, RPN(Region Porposal Network) generates regions of proposal using Convolutional Feature Map calculated by CNN. RPN finds Regions of Interest that may contain an object and extracts that region with help of ROI Pooling layer.

The exctracted regions are then classified by Fast R-CNN. Fast R-CNN Separates object from the background and finally returns the detected bounding box  coordinates of an object.
Besides that, RPN returns “Objectivness Score”, so called Confidence score Which could be used to avoid false positives.

## Image Augmentation

Image Augmentation is done using Imgaug Library.  Faster R-CNN is tend to have many false positives when testing in video. So, Image and Bounding Box Augmentation is one of the method to reduce False Positives. Example:

![augmentation example](https://user-images.githubusercontent.com/82511782/146067347-95784886-26bd-4b22-a1c3-38e630431336.png)

## Non-Maximum Suppression (NMS)

NMS is an algorithm which selects most appropriate bounding box of the object. Object detection algorithms tend to draw many bounding box on single object in the image. With Non-Maximum suppression we suppress less likely bounding boxes and only keep one that has a biggest confidence.


## Camera Calibration and Image Undistortion

Camera Calibratoin calculates  intristic and extristic parameters of camera. With the help of camera matrix we transform real world points to the image plane. 
when camera captures an image, it does not capture the true image but a distortion, in which straight lines are projected as slightly curved ones when perceived through camera lenses. So, objects might appear a little far away from camera than it actually is in real world. So, first thing we do is to remove the distortion from image.
![undistorted](https://user-images.githubusercontent.com/82511782/146067518-7617d989-1901-4efb-ad9e-ef10b024359b.png)

## Birds Eye View 

The idea of birds Eye View is to warp the image, to see the image from above.because it is easier to calculate the line curvature when lane lines seem parallel.
![Birds Eye ](https://user-images.githubusercontent.com/82511782/146067613-8fcfb363-dbb5-440f-b193-a16a3ccf98c2.png)

## Image thresholding

The idea of threhsolding is to process the image such a way that lane lines are seperated from road and are perfectly visible. I have applied different image thresholding method, starting from sobelx and sobely to magnitude threshold and HLS and ect… 

Finaly, after perserving different threhsolidng methods the combination of X and Y sobels are perfect for our video.

One Setback is that in the video, lane lines and other white lines are very close to each other and on some frames it is very difficult to find which is real lane line and which is not. 
![thresholded](https://user-images.githubusercontent.com/82511782/146067746-4337dd64-0507-43f7-ac8c-52c07f9274fc.png)

## Curvature and Lane Line Detection

After applying calibration, thresholding and a perspective transform(birds eye view) we have a binary image where the lane lines stand out clearly. But we still need to decide explicityle which pixels are part of the lines and which belong to the left line and which belong to the right line.
For that, we use Histogram of peaks. We only need the histogram of bottom of the image, since the lane lines most definilty are at the there. Histogram of Peaks identify X and Y coordinates of left and right lane lines.

Second order polynomial is used to fit the lines. On the detected lines, we use sliding window technique non-zero pixels and store that pixels to window. Then we use numpy polyfit to find best second order polynomial to represent the lanes.
![sliding 1 ](https://user-images.githubusercontent.com/82511782/146067869-944caf26-ea4a-4e56-a76e-c2b9fb46a1bb.png)

![sliding 2](https://user-images.githubusercontent.com/82511782/146067884-bc53e777-71a7-4227-91c6-8e9e959e0484.png)

For Curvature we use [This Tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
For the offset to the center of the lane we assume that the camera is mounted exactly in the center of the car. Therefore, it is difference between the center of the image and the middle point of beginning of the lane lines. This returns the radius of curvature. 
The final Results on the images are : 
![final1](https://user-images.githubusercontent.com/82511782/146068084-09a724ea-2bb1-4cf3-8cad-e9375f05a96f.png)

![final2](https://user-images.githubusercontent.com/82511782/146068101-704e72ec-e6f6-44dd-9dbe-c9a4abeae4f3.png)


## Sanity Check
1) Faster R-CNN is very much tend to Too many false positives, Even though i have used NMS, Image Augmentation and predicted only those bounding boxes where Confidence was high, there are still wrong detections in the video.
2) This is a very challenging video. The CV algorithm detects lane lines in different conditions(dark,light). But there is one setback, Like i mentioned before, there is a line very close to lane line that is too hard to seperate from the lane line with histogram. So, i had to do a birds eye view very close to the road that at one frame birds eye view can't see the left lane.  
3) It is preferable that you calibrate camera on at least 20 images.


## Acknowledgments

Inspiration, code snippets, etc.
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [Non-Maximum Suppression](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
* [Image Augmentation](https://github.com/aleju/imgaug)
* [Udacity Self Driving Cars Course](https://www.udacity.com/course/intro-to-self-driving-cars--nd113)
* [Car Dataset](https://www.kaggle.com/sshikamaru/car-object-detection)
