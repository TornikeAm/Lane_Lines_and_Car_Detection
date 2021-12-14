import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import cv2 as cv
import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import warnings
warnings.simplefilter('ignore')
ia.seed(1)


with ZipFile('car_data.zip', 'r') as Zip:
   # Extract all the contents of zip file in different directory
   Zip.extractall('Images')


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.OneOf([
                 iaa.Sequential([
    iaa.Multiply((1.2, 1.5)),
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        scale=(0.5, 0.7)
    )
]),
iaa.Sequential([
    iaa.GammaContrast(1.5),
    iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
]),

iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
]),

iaa.OneOf([
       iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
       iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
       iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
])

df=pd.read_csv('Images/data/train_solution_bounding_boxes (1).csv')
image_names = [name for name in df['image']]
all_bboxes= [df[df['image']==image][['xmin','ymin','xmax','ymax']].values for image in image_names]
conc= dict(zip(image_names,all_bboxes))

path_for_aug='Images/data/training_images/'

class Augmentation:
  def __init__(self,imgbbs,path):
    self.imgbbs=imgbbs
    self.path=path
  
  
  def numOfObjects(self,obj,box,img):
    if obj ==1 :
      bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][2], y2=box[0][3]),
      ], shape=img.shape)
    if obj==2:
       bbs = BoundingBoxesOnImage([
          BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][2], y2=box[0][3]),
          BoundingBox(x1=box[1][0], y1=box[1][1], x2=box[1][2], y2=box[1][3]),
          ], shape=img.shape)
    if obj==3:
      bbs = BoundingBoxesOnImage([
          BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][2], y2=box[0][3]),
          BoundingBox(x1=box[1][0], y1=box[1][1], x2=box[1][2], y2=box[1][3]),
          BoundingBox(x1=box[2][0], y1=box[2][1], x2=box[2][2], y2=box[2][3]),
          ], shape=img.shape)
    
    return bbs
  

  def augment(self):
    print('starting Augmentation')
    print('Augmenting Each image 25 times')
    augmented_images=[]
    x_min,y_min,x_max,y_max=[],[],[],[]
    for image,box in conc.items():
      if len(box)<=3:
        for count in range(25):
          print(f'aug{image[:-4]}{count}.jpg')
          img=imageio.imread(os.path.join(self.path,image))
          
  
  	  bbs= self.numOfObjects(len(box),box,img)
          image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
          
          after_bb = bbs_aug.bounding_boxes[0]
          x_min.append(after_bb.x1)
          y_min.append(after_bb.y1)
          x_max.append(after_bb.x2)
          y_max.append(after_bb.y2)
          
  
          cv.imwrite(f"{self.path}/aug{image[:-4]}{count}.jpg",image_aug)
          augmented_images.append(f"aug{image[:-4]}{count}.jpg")
    #creating new dataframe to pass new images and classes
    ziplists=zip(augmented_images,x_min,y_min,x_max,y_max)
    new_df=pd.DataFrame(ziplists
                        ,columns=['image','xmin','ymin','xmax','ymax'])  
    return new_df

augmented=Augmentation(conc,path_for_aug)
new_df=augmented.augment()

final_df=pd.concat([df,new_df],ignore_index=True)
