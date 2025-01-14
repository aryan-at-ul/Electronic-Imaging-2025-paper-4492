import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, model_vig, model_unet,model_swinunet,modelnewseg
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import os 
from dataset import Dataset
from metrics import *
import pandas as pd
from model import model_smp, model_unet,model_dunet, preprocessing_fn
import segmentation_models_pytorch as smp
from utils import * 
import sys
import torch
from tqdm import tqdm as tqdm
from typing import List
# from focal_loss.focal_loss import FocalLoss

import torch
import numbers
import math
from torch import Tensor, einsum
from torch import nn
from utils import simplex, one_hot
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torch.nn import functional as F
DATA_DIR = 'Dataset_1000'
this_dir_path = os.path.abspath(os.getcwd())


x_train_dir = os.path.join(DATA_DIR , 'images')
y_train_dir = os.path.join(DATA_DIR , 'masks')
# x_train_dir = f"{this_dir_path}/rgb_images"
# y_train_dir = f"{this_dir_path}/semantic_annotations/rgbLabels"
x_valid_dir = os.path.join(DATA_DIR, 'rgb252')
y_valid_dir = os.path.join(DATA_DIR, 'mask252')
x_test_dir = os.path.join(DATA_DIR , 'test50_rgb')
y_test_dir = os.path.join(DATA_DIR , 'test50_mask')

print(len(os.listdir(x_train_dir)))
print(len(os.listdir(y_train_dir)))
print(len(os.listdir(x_valid_dir)))
print(len(os.listdir(y_valid_dir)))
print(len(os.listdir(x_test_dir)))
print(len(os.listdir(y_test_dir)))


print(x_test_dir)

class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()
# class_rgb_dict = {cls: rgb for cls, rgb in zip(class_names, class_rgb_values)}
class_rgb_dict = {cls_name: rgb for cls_name, rgb in zip(class_names, class_rgb_values)}

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


# CLASSES = [ 'road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']
CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']


class_colors_bgr = [
    [0, 0, 0],       # background
    [255, 0, 255],   # road
    [255, 0, 0],     # lanemarks
    [0, 255, 0],     # curb
    [0, 0, 255],     # person
    [255, 255, 255], # rider
    [255, 255, 0],   # vehicles
    [0, 255, 255],   # bicycle
    [128, 128, 255], # motorcycle
    [0, 128, 128]    # traffic sign
]

# Convert BGR to RGB
class_colors_rgb = [list(reversed(color)) for color in class_colors_bgr]

from torch.utils.data import DataLoader


import numpy as np
 
 
class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """
 
    def reset(self):
        """Reset the meter to default settings."""
        pass
 
    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass
 
    def value(self):
        """Get the value of the meter in the current state."""
        pass
 
 
class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0
 
    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n
 
        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))
 
    def value(self):
        return self.mean, self.std
 
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
 

 
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="gpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
 
        self._to_device()
 
    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)
 
    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s
 
    def batch_update(self, x, y):
        raise NotImplementedError
 
    def on_epoch_start(self):
        pass
 
    def run(self, dataloader):
 
        self.on_epoch_start()
 
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
 
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
 
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__class__.__name__: loss_meter.mean}
                logs.update(loss_logs)
 
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
 
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
 
        return logs
 
 
class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
 
    def on_epoch_start(self):
        self.model.train()
        self.model.to(self.device)
 
    def batch_update(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        # loss = self.loss(prediction, y)
        y_class_indices = torch.argmax(y, dim=1)  # Convert to shape [N, H, W] 

        # print("prediction shape: ", prediction.shape)
        
        y_class_indices = y # this is for focal loss only, remove for otehrs (other focal loss, not focal pytorch)
        # y_class_indices = torch.argmax(y, dim=1)  # Convert to [batch_size, height, width] this is for cross entropy, remove when focal

        # prediction = F.softmax(prediction, dim=1) # this is for focal loss only, remove for otehrs
        # print("prediction shapea after softmax: ", prediction.shape)
        # print("y shape: ", y.shape)
        loss = self.loss(prediction,y_class_indices)
        # loss = self.loss(prediction,y_class_indices,_)
        loss.backward()
        self.optimizer.step()
        return loss, prediction
 
 
class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )
 
    def on_epoch_start(self):
        self.model.eval()
        self.model.to(self.device)
 
    def batch_update(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            prediction = self.model.forward(x)
            # loss = self.loss(prediction, y)
            y_class_indices = torch.argmax(y, dim=1)
            y_class_indices = y # this is for focal loss only, remove for otehrs (other focal loss, not focal pytorch)
            # prediction = F.softmax(prediction, dim=1) # this is for focal loss only, remove for otehrs

            loss = self.loss(prediction, y_class_indices)

        return loss, prediction



train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
 
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# model_dunet model_vig= model_def_dunet#model_unet # keep changing this 


# optimizer = torch.optim.Adam([ 
#     dict(params=model_dunet.parameters(), lr=0.0001),
# ])


class_indices = {class_name: index for index, class_name in enumerate(CLASSES)}


# classes_to_consider = ['road', 'person', 'vehicles']
classes_to_consider = ['background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']   
print(class_indices)
idc = [class_indices[class_name] for class_name in classes_to_consider]


class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"
    def __init__(self, class_weights: Tensor, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        alpha = 0.15
        gamma = 3
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.class_weights = class_weights.cuda()

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        BCE_loss = F.binary_cross_entropy_with_logits(pc, tc, reduction='none')
        # print("BCE_loss shape:", BCE_loss.shape)

        targets = tc.type(torch.long)
        pt = torch.exp(-BCE_loss)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = at.view_as(tc)
        F_loss = (1 - pt) ** self.gamma * BCE_loss * at
        # print("F_loss shape:", F_loss.shape)
        class_weights = self.class_weights.view([1, -1, 1, 1])
        class_weights = class_weights.expand_as(F_loss)

        # print("class_weights shape:", class_weights.shape)  if not dim 4 life is unfair!! 
        weighted_loss = class_weights * F_loss

        return weighted_loss.mean()
 




class Focal_Loss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        alpha=.15
        gamma=3
        super(Focal_Loss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        #self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def __call__(self,probs: Tensor, target: Tensor) -> Tensor:
        # pc = probs[:, self.idc, ...].type(torch.float32)
        # tc = target[:, self.idc, ...].type(torch.float32)
        # print("probs shape: ", probs.shape)
        # print("target shape: ", target.shape)
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        BCE_loss = F.binary_cross_entropy_with_logits(pc, tc, reduction='none')
        targets = tc.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class DiceLossC(torch.nn.Module):
    def __init__(self, eps=1e-6, beta=1.0, activation='sigmoid', ignore_channels=None):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.activation = activation
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
 
        if self.activation == 'sigmoid':
            y_pr = torch.sigmoid(y_pr)
        elif self.activation == 'softmax':
            y_pr = torch.softmax(y_pr, dim=1)
        elif self.activation is not None:
            raise NotImplementedError(f"Activation '{self.activation}' is not implemented")


        return 1 - self.f_score(y_pr, y_gt)

    def f_score(self, y_pr, y_gt):

        y_pr = y_pr.contiguous().view(-1)
        y_gt = y_gt.contiguous().view(-1)


        intersection = (y_pr * y_gt).sum()
        fp = (y_pr * (1 - y_gt)).sum()
        fn = ((1 - y_pr) * y_gt).sum()

        numerator = (1 + self.beta**2) * intersection + self.eps
        denominator = (1 + self.beta**2) * intersection + self.beta**2 * fn + fp + self.eps
        return numerator / denominator



loss = DiceLossC()


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# num_params_vig = count_parameters(model_dunet)
# print("Model ViG Parameter Count:", num_params_vig)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
) 
 
# test_dataloader = DataLoader(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

 
metrics = [
    IoU(threshold=0.5),
    Accuracy(threshold=0.5),
    Fscore(threshold=0.5),
    Recall(threshold=0.5),
    Precision(threshold=0.5),
]



 

import os
import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

results_dir = "wood_result_unet"
save_prediction_dir = results_dir
os.makedirs(results_dir, exist_ok=True)


# model =  model_vig
# model = model_unet
# model =  model_swinunet
model = modelnewseg


# Load the trained model
# model = torch.load("swin_wood.pth")
model = torch.load("graphbottleneck_wood.pth")
# model = torch.load("unet_wood.pth")
# model = torch.load("swin_wood.pth")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decode_segmentation_mask(mask, class_colors):
    """
    mask: H x W with class indices
    class_colors: list of RGB tuples
    Returns an H x W x 3 color image
    """
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_index, color in enumerate(class_colors):
        color_image[mask == cls_index] = color
    return color_image


with torch.no_grad():
    for i, (image, _) in enumerate(test_loader):
        image = image.to(device)  # shape: [B, C, H, W]
        print("image shape: ", image.shape) 
        # Forward pass
        output = model(image)  # output shape: [B, C, H, W]
        print("output shape: ", output.shape)
        
        # Get predicted class for each pixel
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # shape: [H, W]
        print("pred_mask shape: ", pred_mask.shape)
        print("pred_mask unique values: ", np.unique(pred_mask))

        # Decode prediction
        color_mask = decode_segmentation_mask(pred_mask, class_colors_rgb)

        # Get the original image file name for saving
        img_path = test_dataset.images_fps[i]
        img_name = os.path.basename(img_path)
        
        # Load original image
        orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Load original mask (ground truth)
        # Assuming test_dataset.masks_fps contains the corresponding mask paths
        gt_mask_path = test_dataset.masks_fps[i]
        # gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        
        gt_mask_bgr = cv2.imread(gt_mask_path, cv2.IMREAD_COLOR)
        gt_mask_rgb = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2RGB)

        # Convert color-coded mask to class indices using the same approach as the dataset:
        gt_class_mask = np.zeros((gt_mask_rgb.shape[0], gt_mask_rgb.shape[1]), dtype=np.uint8)
        for idx, color in enumerate(class_colors_rgb):
            gt_class_mask[np.all(gt_mask_rgb == color, axis=-1)] = idx

        # Now gt_class_mask is a 2D array of class indices. 
        # Decode it to get a colored mask for visualization:
        gt_color_mask = decode_segmentation_mask(gt_class_mask, class_colors_rgb)

        # Resize both to the original image size if needed
        orig_h, orig_w = orig_img.shape[:2]
        gt_color_mask_resized = cv2.resize(gt_color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        pred_color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Plot all three: original image, ground truth mask, predicted mask
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(orig_img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Original Mask")
        plt.imshow(gt_color_mask_resized)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_color_mask_resized)
        plt.axis('off')

        plt.show()