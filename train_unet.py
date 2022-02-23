
'''
# Commented out IPython magic to ensure Python compatibility.
# Needed libs
# %pip install poutyne
# %pip install segmentation-models-pytorch
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
!pip uninstall albumentations
!pip install --upgrade albumentations
!git clone https://github.com/matterport/Mask_RCNN
!pip install -U cloths_segmentation

!pip install --upgrade opencv-python
!pip install --upgrade opencv-contrib-python
'''

import os
from datetime import datetime, timedelta
from pathlib import Path as getFileName
import glob
import sys
import xml.etree.ElementTree as ET
import random
import shutil

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch

os.chdir('Mask_RCNN/samples')
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn import visualize as mrcnn_funcs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

COLORS = [(1.0, 0.0, 0.0)]
CLASSES=['stressed']
SEED = 42

VALID_FOLDERS_IMGS = ['/content/f/JPEG_5_S/', '/content/f/JPEG_9_S/', '/content/f/JPEG_10_S/']
VALID_FOLDERS_MASKS = ['/content/JPEG_10_S_masks/', '/content/JPEG_9_S_masks/', '/content/JPEG_5_S_masks/']

TRAIN_FOLDERS_IMGS = ['/content/f/JPEG/', '/content/f/JPEG_3/', '/content/f/JPEG_6/', '/content/f/JPEG_8/', '/content/f/JPEG_16_F/']
TRAIN_FOLDERS_MASKS = ['/content/JPEG_8_masks/', '/content/JPEG_6_masks/', '/content/JPEG_3_masks/', '/content/JPEG_masks/', '/content/JPEG_16F_masks/']

SICK_FOLDERS = ['/content/f/JPEG_16_F/', '/content/f/JPEG_5_S/', '/content/f/JPEG_9_S/', '/content/f/JPEG_10_S/']
HEALTHY_FOLDERS = ['/content/f/JPEG/', '/content/f/JPEG_3/', '/content/f/JPEG_6/', '/content/f/JPEG_8/']


SICK_IMAGES = [glob.glob(dir+'*.jpg') for dir in SICK_FOLDERS]
SICK_IMAGES = [j for i in SICK_IMAGES for j in i]
SICK_IMAGES = [getFileName(img).stem for img in SICK_IMAGES]
HEALTHY_IMAGES = [glob.glob(dir+'*.jpg') for dir in HEALTHY_FOLDERS]
HEALTHY_IMAGES = [j for i in HEALTHY_IMAGES for j in i]
HEALTHY_IMAGES = [getFileName(img).stem for img in HEALTHY_IMAGES]

CROP_IMG_WIDTH = 1000
CROP_IMG_LENGTH = 1000
ORIG_IMG_WIDTH = 3648
ORIG_IMG_LENGTH = 2736
IMG_WIDTH = 512
IMG_LENGTH = 512

DIR_TRAIN_IMGS = '/content/train_imgs/'
DIR_TRAIN_MASKS = '/content/train_masks/'

DIR_VALID_IMGS = '/content/valid_imgs/'
DIR_VALID_MASKS = '/content/valid_masks/'

def make_valid_folder(VALID_FOLDERS_IMGS, VALID_FOLDERS_MASKS):
  os.mkdir(DIR_VALID_IMGS)
  os.mkdir(DIR_VALID_MASKS)

  for dir_ in [glob.glob(dir+'*.jpg') for dir in VALID_FOLDERS_IMGS]:
    for file_name in dir_:
        print(file_name)
        shutil.copy(file_name, DIR_VALID_IMGS)

  for dir_ in [glob.glob(dir+'*.jpg') for dir in VALID_FOLDERS_MASKS]:
    for file_name in dir_:
        shutil.copy(file_name, DIR_VALID_MASKS)
  
  to_c = [glob.glob(dir+'*.jpg') for dir in VALID_FOLDERS_IMGS]
  assert(len([j for i in to_c for j in i]) == len(glob.glob('/content/valid_imgs/'+'*.jpg')) )

def make_train_folder(TRAIN_FOLDERS_IMGS, TRAIN_FOLDERS_MASKS):
  os.mkdir(DIR_TRAIN_IMGS)
  os.mkdir(DIR_TRAIN_MASKS)

  for dir_ in [glob.glob(dir+'*.jpg') for dir in TRAIN_FOLDERS_IMGS]:
    for file_name in dir_:
        #print(file_name)
        shutil.copy(file_name, DIR_TRAIN_IMGS)

  for dir_ in [glob.glob(dir+'*.jpg') for dir in TRAIN_FOLDERS_MASKS]:
    for file_name in dir_:
        #print(type(file_name), type(DIR_TRAIN_MASKS))
        shutil.copy(file_name, DIR_TRAIN_MASKS)
  
  to_c = [glob.glob(dir+'*.jpg') for dir in TRAIN_FOLDERS_IMGS]
  assert(len([j for i in to_c for j in i]) == len(glob.glob('/content/train_imgs/'+'*.jpg')) )

make_valid_folder(VALID_FOLDERS_IMGS, VALID_FOLDERS_MASKS)
make_train_folder(TRAIN_FOLDERS_IMGS, TRAIN_FOLDERS_MASKS)

print(len(SICK_IMAGES), len(HEALTHY_IMAGES), len(glob.glob(DIR_TRAIN_IMGS+'*.jpg')))
print(len(glob.glob(DIR_TRAIN_MASKS+'*.jpg')))

print(len(SICK_IMAGES), len(glob.glob(DIR_VALID_IMGS+'*.jpg')))
print(len(glob.glob(DIR_VALID_MASKS+'*.jpg')))

"""# New Section"""

def visualize(**images):
  n = len(images)
  plt.figure(figsize=(16, 5))
  for i, (name, image) in enumerate(images.items()):
      plt.subplot(1, n, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.title(' '.join(name.split('_')).title())
      plt.imshow(image)
  plt.show()

def plot_any(arr):
  plt.figure(figsize=(16, 10))
  plt.imshow(arr)
  plt.show()

def overlay_mask_multiple(image, mask):
  copied_img = np.copy(image)
  for i in range(len(CLASSES)):
    mask_class = mask[:, :, i].squeeze().astype(int)
    copied_img = mrcnn_funcs.apply_mask(copied_img, mask_class, COLORS[i])
  plot_any(copied_img)

def overlay_mask_single(image, mask, idx=0):
  copied_img = np.copy(image)
  mask_class = (mask==idx).astype(int)
  copied_img = mrcnn_funcs.apply_mask(copied_img, mask_class, COLORS[idx])
  plot_any(copied_img)

def get_cmap(arr, cmap='jet'):
  cm = plt.get_cmap(cmap)
  return cm(arr)

def plotNDVI(arr):
  arr = get_cmap(arr)
  plot_any(arr)

class analyze_image:

  def __init__(self, path):
    self.path = path
    self.img = plt.imread(self.path)
    self.NDVI = self.turnIntoNDVI(self.img)
    self.xml_path = glob.glob(str(getFileName(self.path).parent) + '/*.xml')
    print(self.xml_path)
  
  def turnIntoNDVI(self, arr):
    _, R, NearIR = cv2.split(arr.astype(np.float))
    NDVI = (.25*(NearIR - R)) / (.25*(NearIR + R + 0.01))
    new_img = NDVI+.45*(arr[:, :, 2]/255.)
    new_img = new_img/(1.45)
    return new_img

  def remove_background(self, arr):
    blurImg = cv2.GaussianBlur(arr, (5,5), 0)   
    hsv = cv2.cvtColor(blurImg, cv2.COLOR_RGB2HSV)  
    lower_hsv = np.array([147, 0, 106])
    higher_hsv = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0 
    clear = np.zeros_like(arr, np.uint8)
    clear[bMask] = arr[bMask] 
    return clear
  
  def getBoxes(self, xml_pth):
    tree = ET.parse(xml_pth)
    root = tree.getroot()
    objects = root.findall('object')
    box_cords = []
    for o in objects:
      box = o.find('bndbox')
      xMax = int(box.find('xmax').text)
      yMax = int(box.find('ymax').text)
      xMin = int(box.find('xmin').text)
      yMin = int(box.find('ymin').text)
      box_cords.append([xMax, yMax, xMin, yMin])
    return box_cords
  
  def box_mask(self, shape, vertices):
    arr = np.ones(shape[:-1])
    for box in vertices:
        xMax, yMax, xMin, yMin= box
        arr[yMin:yMax, xMin:xMax] = 0
    return arr
  
  def getPolyCords(self, xml_pth):
    tree = ET.parse(xml_pth)
    root = tree.getroot()
    objects = root.findall('object')
    box_cords = []
    for o in objects:
      box = o.find('polygon')
      x_and_y_points = list(box.iter())
      polygon_list = []
      for i in range(1, len(x_and_y_points), 2):
        x, y = int(x_and_y_points[i].text) , int(x_and_y_points[i+1].text)
        polygon_list.append((x, y))
      box_cords.append(polygon_list)
    return box_cords
  
  def poly_mask(self, shape, *vertices):
      height, width = shape[:-1]
      img = Image.new(mode='L', size=(width, height), color=1)
      draw = ImageDraw.Draw(img)
      for polygon in vertices:
        draw.polygon(polygon, outline=0, fill=0)
      mask = np.array(img).astype('float32')
      return mask
  
  def makeClassesFromNDVI(self, thres=.6):
    arr = np.copy(self.NDVI)
    
    if len(self.xml_path) != 0:
      
      try:
        poly_cords = self.getPolyCords(self.xml_path[0])
        poly_mask = self.poly_mask(self.img.shape, *poly_cords)
        arr[np.where(poly_mask == 0)] = CLASSES.index('stressed')
      except Exception as e:
        print(e)
        box_cords = self.getBoxes(self.xml_path[0])
        box_mask = self.box_mask(self.img.shape, box_cords)
        arr[np.where(box_mask == 0)] = CLASSES.index('stressed')

      new_img_without_back = self.remove_background(self.img)
      arr[np.sum(new_img_without_back/255., axis=2)==0] = 3
      
      arr_non = np.where((self.NDVI>thres), 3, arr)
      arr[arr_non==3] = 3
    
    return arr


"""# New Section"""

def findTMinusIMG(DIR, path, t, range=15):
  files = glob.glob(DIR+'*.*')
  dt_of_path = datetime.strptime(getFileName(path).stem, '%d_%m_%Y__%H_%M%p')
  t_minus = dt_of_path - timedelta(minutes = t)
  start = t_minus - timedelta(minutes = range)
  end = t_minus + timedelta(minutes = range)
  
  for file in files:
    dt_of_file = datetime.strptime(getFileName(file).stem, '%d_%m_%Y__%H_%M%p')
    if start <= dt_of_file <= end:
      return file

class Dataset(BaseDataset):

    def __init__(self, images_dir, mask_dir, classes=CLASSES, augmentation=None, preprocessing=None,):
      
        self.images_dir = images_dir
        self.mask_dir = mask_dir 
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.EXT = '*.jpg'
        
        self.t_minus = 60
        self.images_fps = []
        for file in glob.glob(self.images_dir+ self.EXT):
          if findTMinusIMG(self.images_dir, file, self.t_minus) is not None:
            self.images_fps.append(file)
          else:
            #dt_of_path = datetime.strptime(getFileName(file).stem, '%d_%m_%Y__%H_%M%p')
            #print('Did not have a file', self.t_minus, 'minutes before:', dt_of_path)
            pass

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.axis = -1 if self.preprocessing==None else 0
    
    def __getitem__(self, i):
        #random.seed(i)

        image_0M = cv2.imread(self.images_fps[i])
        image_60M = cv2.imread(findTMinusIMG(self.images_dir, self.images_fps[i], self.t_minus))
        image_0M = cv2.cvtColor(image_0M, cv2.COLOR_BGR2RGB)
        image_60M = cv2.cvtColor(image_60M, cv2.COLOR_BGR2RGB)
        #print(np.max(image_0M), np.max(image_60M), np.min(image_0M), np.min(image_60M))
        
        mask_path  = getFileName(self.images_fps[i]).stem + self.EXT[1:]
        mask = plt.imread(self.mask_dir + mask_path)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image_0M, image2=image_60M, mask=mask)
            image_0M, image_60M, mask = sample['image'], sample['image2'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image_0M, image2=image_60M, mask=mask)
            image_0M, image_60M, mask = sample['image'], sample['image2'], sample['mask']
            #print(image_0M.shape, image_60M.shape, mask.shape, np.max(image_0M), np.max(image_60M), np.min(image_0M), np.min(image_60M))
            
        image = np.concatenate((image_60M, image_0M), axis=self.axis)
        return image, mask
        
        '''
        if getFileName(path).stem in SICK_IMAGES:
          arr_of_0_or_1 = 1
        if getFileName(path).stem in HEALTHY_IMAGES:
          arr_of_0_or_1 =  0
        return np.zeros((5,5)),arr_of_0_or_1
        '''
    def __len__(self):
        return len(self.images_fps)

"""# New Section"""

from torch.utils.data import WeightedRandomSampler

def make_sampler(dir_of_chosen, weights=[.7, .3], seed=SEED):
  target = []
  for pth_c_imgs in dir_of_chosen:
    if getFileName(pth_c_imgs).stem in SICK_IMAGES:
      target.append(1)
    if getFileName(pth_c_imgs).stem in HEALTHY_IMAGES:
      target.append(0)
  assert(len(target) == len(dir_of_chosen))

  class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])
  print('SAMPLE COUNT FOR EACH CLASS:', class_sample_count)

  weight = (weights / class_sample_count)
  samples_weight = np.array([weight[t] for t in target])
  samples_weight = torch.from_numpy(samples_weight)
  samples_weight = samples_weight.double()
  return WeightedRandomSampler(samples_weight, len(samples_weight),  generator=torch.Generator().manual_seed(seed))

"""# New Section"""

import albumentations as albu

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_training_augmentation_medium(): # 2048 * 1536
    train_transform = [
                      albu.Resize(2736,3648,always_apply=True),
                      albu.OneOf([
                                  albu.Crop(x_min=1500, y_min=0, x_max=3648, y_max=1500, always_apply=True),
                                  albu.Crop(x_min=1500, y_min=1236, x_max=3648, y_max=2736, always_apply=True)
                                  ], p=1 ),
                       albu.RandomCrop(height=1500, width=1500, always_apply=True),
                       albu.Resize(512,512,always_apply=True),
                       
                       albu.HorizontalFlip(p=0.5),
                       albu.VerticalFlip(p=0.5),
                       albu.RandomRotate90(p=.5),
                       albu.Transpose(p=.5)
    ]
    return albu.Compose(train_transform, additional_targets={'image': 'image', 'image2':'image'})


def get_validation_augmentation():
    test_transform = [
        albu.Resize(2736,3648,always_apply=True),
        albu.Crop(x_min=0, y_min=0, x_max=1500, y_max=2736, always_apply=True),
        albu.OneOf([
               albu.Crop(x_min=0, y_min=0, x_max=1500, y_max=1500, always_apply=True),
               albu.Crop(x_min=0, y_min=1236, x_max=1500, y_max=2736, always_apply=True)
               ], p=1 ),
        #albu.RandomCrop(height=1500, width=1500, always_apply=True),
        albu.Resize(512,512,always_apply=True),
    ]
    return albu.Compose(test_transform, additional_targets={'image': 'image', 'image2':'image'})


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, additional_targets={'image': 'image', 'image2':'image'})


"""# New Section"""

# set up good validation -> augs -> weights for loss function -> increase batch size and adjust learning rate / scheduler -> workers

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss, DiceLoss, FocalLoss, BalancedBCEWithLogitsLoss, SoftBCEWithLogitsLoss
import torchmetrics
from cloths_segmentation.metrics import binary_mean_iou
import wandb

class ImagePredictionLogger(pl.Callback):
    def __init__(self, img_loader):
        super().__init__()
        self.img_loader = img_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        
        batch = next(iter(self.img_loader))
        
        val_imgs, val_masks = batch
        logits = pl_module(val_imgs.to(device=pl_module.device))
        val_imgs, val_masks, logits = val_imgs.cpu(), val_masks.cpu(), logits.cpu()

        imgs_to_plot = [np.concatenate((img[3:, :, :], 
                                        np.repeat(gt_mask, 3, axis=0), 
                                        np.repeat(pred_mask, 3, axis=0)), axis=-1) for img, gt_mask, pred_mask in zip(val_imgs, val_masks, logits)]
      
        trainer.logger.experiment.log({
            "example_img": [wandb.Image(np.moveaxis(plot_img, 0, -1))
                            for plot_img in imgs_to_plot],
            "global_step": trainer.global_step
            })

class SaveModelWeights(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
      torch.save(pl_module.model.state_dict(), '/content/model_{}.pth'.format(trainer.current_epoch))

class SegmentVegetation(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        
        #pl.seed_everything(SEED, workers=True)
        
        self.model = model

        self.losses = [
            ("dice", 0.1, DiceLoss(mode="binary", from_logits=True)),
            ("bce", 0.3, BalancedBCEWithLogitsLoss()),
            ("focal", 0.6, BinaryFocalLoss()),
        ]
        

    def forward(self, batch: torch.Tensor) -> torch.Tensor: 
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([  
                                      dict(params=self.model.parameters(), lr=.00005, weight_decay=.0001,)
                                      ])
        self.optimizers = [optimizer]
        #lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizers[0], mode='max', factor=.1, patience=2, verbose=True),
        #                 "monitor": "val_loss"}

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
        #                                                                 T_max=8,
        #                                                                 eta_min=1e-6,
        #                                                                 verbose=True,)
        return self.optimizers

    def training_step(self, batch, batch_idx):
        features, masks = batch
        logits = self.forward(features)
        total_loss = 0
        train_iou_step = binary_mean_iou(logits, masks)
        logs = {}
        
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            logs[f"train_mask_{loss_name}"] = ls_mask
        
        logs["train_loss"] = total_loss
        logs["lr"] = self._get_current_lr()
        return {"loss": total_loss, "log": logs, "train_iou": train_iou_step}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id):
        features, masks = batch
        logits = self.forward(features)
        result = {}
        total_loss = 0

        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            result[f"val_mask_{loss_name}"] = ls_mask

        result["validation_loss"] = total_loss
        result["val_iou"] = binary_mean_iou(logits, masks)
        return result
    
    def training_epoch_end(self, outputs):
      logs = {"epoch": self.trainer.current_epoch}
      avg_train_iou = find_average(outputs, "train_iou")
      logs["train_iou"] = avg_train_iou
      print(logs)
    
    def validation_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch}
        
        avg_val_iou = find_average(outputs, "val_iou")
        logs["val_iou"] = avg_val_iou

        avg_val_loss = find_average(outputs, "validation_loss")
        logs["val_loss"] = avg_val_loss

        self.log('val_iou', avg_val_iou)
        self.log('val_loss', avg_val_loss)

        print(logs)
        return {"val_iou": avg_val_iou, "log": logs}

def main():
    pl.seed_everything(SEED, workers=True)

    model_flags = {'ENCODER': 'se_resnext50_32x4d', # se_resnext101_32x4d # timm-efficientnet-b5'
                    'ENCODER_WEIGHTS': 'imagenet',
                    'CLASSES': 1,
                    'IN_CHANNELS': 6,
                    'ATTENTION': 'scse'}

    data_flags={'TRAIN_IMGS': '/content/train_imgs/',
                'TRAIN_MASKS':'/content/train_masks/',
                'VALID_IMGS': '/content/valid_imgs/',
                'VALID_MASKS': '/content/valid_masks/',
                'TRAIN_AUG': get_training_augmentation_medium(),
                'VALID_AUG': get_validation_augmentation(),
                }

    sampler_flags = {'LABEL_WEIGHTS': [.5, .5]} 

    loader_flags = {'TRAIN_BATCH_SIZE': 2,
                    'VALID_BATCH_SIZE': 2,
                    'WORKERS_TRAIN': 0,
                    'WORKERS_VALID': 0}

    trainer_flags = {'EPOCHS': 25}

    model = smp.UnetPlusPlus(
        encoder_name=model_flags['ENCODER'], 
        encoder_weights=None,
        decoder_attention_type=model_flags['ATTENTION'],
        in_channels=model_flags['IN_CHANNELS'], 
        classes=model_flags['CLASSES'])
    
    checkpoint = torch.load('/content/drive/MyDrive/model_13.pth')
    model.load_state_dict(checkpoint)

    ''' MAKE DATASETS '''
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_flags['ENCODER'], model_flags['ENCODER_WEIGHTS'])
    train_dataset = Dataset(
        images_dir=data_flags['TRAIN_IMGS'], 
        mask_dir=data_flags['TRAIN_MASKS'],
        augmentation=data_flags['TRAIN_AUG'], 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    valid_dataset = Dataset(
        images_dir=data_flags['VALID_IMGS'],
        mask_dir=data_flags['VALID_MASKS'],
        augmentation=data_flags['VALID_AUG'], 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    sampler = make_sampler(train_dataset.images_fps, weights=sampler_flags['LABEL_WEIGHTS'])
    ''' MAKE LOADERS '''
    train_loader = DataLoader(train_dataset, 
                              batch_size=loader_flags['TRAIN_BATCH_SIZE'], 
                              num_workers=loader_flags['WORKERS_TRAIN'], 
                              sampler=sampler, 
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=loader_flags['VALID_BATCH_SIZE'], 
                              num_workers=loader_flags['WORKERS_VALID'], 
                              shuffle=False, pin_memory=True, drop_last=False)
    imgLogger_loader =  DataLoader(valid_dataset, 
                            batch_size=loader_flags['VALID_BATCH_SIZE'], 
                            num_workers=loader_flags['WORKERS_VALID'], 
                            shuffle=False, pin_memory=True, drop_last=False)

    #wandb_logger = WandbLogger(project='CROP HEALTH MAPS')

    pipeline = SegmentVegetation(model)

    trainer = pl.Trainer(max_epochs=trainer_flags['EPOCHS'],
                         gpus=1,
                         benchmark=True,
                         #precision=16, 
                         gradient_clip_val=0,
                         default_root_dir="/content/", 
                         callbacks=[SaveModelWeights()],
                         deterministic=True
                         )
                          #logger=wandb_logger,
                          #callbacks=[ImagePredictionLogger(imgLogger_loader)],)

    trainer.fit(pipeline, train_loader, valid_loader,)


main()
