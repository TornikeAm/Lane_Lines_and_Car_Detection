import cv2 as cv
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import os 
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
from Augmentation import final_df


class carData(Dataset):
  def __init__(self,image_dir,df):
    self.df=df
    self.dir=image_dir
    self.image_ids=self.df['image'].unique()

  
  def __getitem__(self,index):
    image_id=self.image_ids[index]
    bboxes=self.df[self.df['image']==image_id]

    img_path=os.path.join(self.dir,image_id)
    image=cv.imread(img_path,cv.IMREAD_COLOR)
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB).astype(np.float32)
    image /=255.0

    boxes=bboxes[['xmin','ymin','xmax','ymax']].values

    area= (boxes[:,3] -boxes[:,1]) * (boxes[:,2] - boxes[:,0])
    
    boxes=torch.as_tensor(boxes,dtype=torch.float32)
    area=torch.as_tensor(area,dtype=torch.float32)

    labels=torch.ones((bboxes.shape[0],),dtype=torch.int64)
    iscrowd=torch.zeros((bboxes.shape[0],),dtype=torch.int64)

    target={}
    target['boxes']=boxes
    target['labels']=labels
    target['image_id'] = torch.tensor([index])
    target['area']=area
    target['iscrowd'] = iscrowd

    image=torchvision.transforms.ToTensor()(image)
    image=image.permute(1,2,0)

    
    return image,target

  def __len__(self):
    return self.image_ids.shape[0]

dataset=carData('Images/data/training_images',final_df)


def collate_fn(batch):
    return tuple(zip(*batch))

train_set,val_set=torch.utils.data.random_split(dataset,[round(len(dataset)*0.9),round(len(dataset)*0.1)])

train_data_loader = DataLoader(
    dataset,
    batch_size=9,
    shuffle=True,
    collate_fn=collate_fn
)


valid_data_loader = DataLoader(
    dataset,
    batch_size=6,
    shuffle=False,
    collate_fn=collate_fn
)
