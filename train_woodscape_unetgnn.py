import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, model_vig, model_unet
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
from lovasz_losses import * 
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
        
        # y_class_indices = y # this is for focal loss only, remove for otehrs (other focal loss, not focal pytorch)
        # y_class_indices = torch.argmax(y, dim=1)  # Convert to [batch_size, height, width] this is for cross entropy, remove when focal
        # y_class_indices = y_class_indices.long()

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
            # y_class_indices = y # this is for focal loss only, remove for otehrs (other focal loss, not focal pytorch)
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
model_dunet =  model_vig
# model_dunet = model_unet
model = model_vig
optimizer = torch.optim.Adam([ 
    dict(params=model_dunet.parameters(), lr=0.0001),
])


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
 


import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        """
        alpha: Can be a float or a list/array of length C with per-class weights.
               If None, all classes are treated equally.
        gamma: Focusing parameter.
        reduction: 'mean', 'sum', or 'none'.
        """
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            # If alpha is a single float value, we use it for the minority class and (1 - alpha) for others
            # But for multi-class, it's better to provide a list or tensor of per-class alphas.
            if isinstance(alpha, (float, int)):
                # If a single alpha is given, assume uniform alpha for all classes
                self.alpha = [alpha]
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits, target):
        """
        logits: [B, C, H, W] (raw, unnormalized predictions from the model)
        target: [B, H, W] with class indices in [0, C-1]
        """
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]

        # Gather the log probabilities corresponding to the ground-truth class
        # target.unsqueeze(1) -> [B, 1, H, W] so gather along class dimension
        log_p = log_probs.gather(1, target.unsqueeze(1))  # [B, 1, H, W]
        log_p = log_p.squeeze(1)  # [B, H, W]
        
        # Convert log_p to p
        p = log_p.exp()  # p is the probability of the correct class
        
        # If alpha is provided, apply class-specific weighting
        if self.alpha is not None:
            # Move alpha to device
            device = logits.device
            alpha_t = torch.tensor(self.alpha, dtype=log_p.dtype, device=device)
            # Gather alpha values per-pixel based on target class
            alpha_factor = alpha_t[target]  # [B, H, W]
        else:
            alpha_factor = 1.0
        
        focal_term = (1 - p) ** self.gamma
        loss = -alpha_factor * focal_term * log_p  # [B, H, W]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



# class Focal_Loss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, **kwargs):
#         self.idc: List[int] = kwargs["idc"]
#         alpha=.15
#         gamma=3
#         super(Focal_Loss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         #self.alpha = torch.tensor([alpha, 1-alpha])
#         self.gamma = gamma

#     def __call__(self,probs: Tensor, target: Tensor) -> Tensor:
#         # pc = probs[:, self.idc, ...].type(torch.float32)
#         # tc = target[:, self.idc, ...].type(torch.float32)
#         # print("probs shape: ", probs.shape)
#         # print("target shape: ", target.shape)
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         tc = target[:, self.idc, ...].type(torch.float32)

#         BCE_loss = F.binary_cross_entropy_with_logits(pc, tc, reduction='none')
#         targets = tc.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = (1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()


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

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLossMultiClass(nn.Module):
    def __init__(self, eps=1e-6, beta=1.0, activation='softmax'):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.activation = activation

    def forward(self, y_pr, y_gt):
        # y_pr: [B, C, H, W], logits
        # y_gt: [B, H, W], integer class indices OR [B, C, H, W] if already one-hot

        if self.activation == 'sigmoid':
            # Use this only for binary (C=1) or multi-label (C>1) tasks
            y_pr = torch.sigmoid(y_pr)
        elif self.activation == 'softmax':
            # For multi-class (C>1) tasks
            y_pr = F.softmax(y_pr, dim=1)

        # If y_gt is not one-hot, convert it
        if y_gt.dim() == 3:  
            # y_gt: [B, H, W], convert to [B, C, H, W]
            B, H, W = y_gt.shape
            C = y_pr.shape[1]
            y_gt_one_hot = torch.zeros((B, C, H, W), device=y_gt.device)
            y_gt_one_hot.scatter_(1, y_gt.unsqueeze(1), 1)
        else:
            y_gt_one_hot = y_gt

        return 1 - self.f_score(y_pr, y_gt_one_hot)

    def f_score(self, y_pr, y_gt_one_hot):
        # y_pr, y_gt_one_hot: [B, C, H, W]

        # Compute per-channel Dice
        intersection = (y_pr * y_gt_one_hot).sum(dim=(0, 2, 3))  # sum over batch,H,W
        fp = (y_pr * (1 - y_gt_one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - y_pr) * y_gt_one_hot).sum(dim=(0, 2, 3))

        numerator = (1 + self.beta**2) * intersection + self.eps
        denominator = (1 + self.beta**2) * intersection + self.beta**2 * fn + fp + self.eps

        dice_per_class = numerator / denominator
        return dice_per_class.mean()  # average over classes


class CombinedDiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0, eps=1e-6, beta=1.0, activation='softmax'):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLossMultiClass(eps=eps, beta=beta, activation=activation)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_pr, y_gt):
        # y_pr: [B, C, H, W], logits
        # y_gt: [B, H, W], integer class indices

        # Dice loss computation
        dice_loss_value = self.dice_loss(y_pr, y_gt)

        # Cross-Entropy loss computation
        ce_loss_value = self.ce_loss(y_pr, y_gt)

        # Combined loss
        combined_loss = self.dice_weight * dice_loss_value + self.ce_weight * ce_loss_value

        return combined_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDiceLovaszLoss(nn.Module):
    def __init__(
        self,
        dice_weight=1.0,
        lovasz_weight=1.0,
        eps=1e-6,
        beta=1.0,
        activation='softmax',
        classes='all',  # Use 'all' to include all 10 classes in the loss
        per_image=False,  # Set to True to compute loss per image
        ignore=None
    ):
        super(CombinedDiceLovaszLoss, self).__init__()
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.dice_loss = DiceLossMultiClass(eps=eps, beta=beta, activation=activation)
        self.lovasz_loss = lovasz_softmax  # Ensure this is correctly imported
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, y_pr, y_gt):
        # Convert one-hot to class indices if necessary
        if y_gt.dim() == 4:
            y_gt = torch.argmax(y_gt, dim=1)

        # Dice Loss
        dice_loss_value = self.dice_loss(y_pr, y_gt)

        # Lovász-Softmax Loss
        probas = F.softmax(y_pr, dim=1)
        lovasz_loss_value = self.lovasz_loss(
            probas,
            y_gt,
            classes=self.classes,
            per_image=self.per_image,
            ignore=self.ignore
        )

        # Combined Loss
        combined_loss = self.dice_weight * dice_loss_value + self.lovasz_weight * lovasz_loss_value

        return combined_loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure Lovász-Softmax functions are correctly imported
# from lovasz_losses import lovasz_softmax

class CombinedDiceLovaszCELoss(nn.Module):
    def __init__(
        self,
        dice_weight=1.0,
        lovasz_weight=1.0,
        ce_weight=1.0,
        eps=1e-6,
        beta=1.0,
        activation='softmax',
        classes='all',
        per_image=False,
        ignore=None
    ):
        """
        Combines Dice Loss, Cross Entropy Loss, and Lovász-Softmax Loss.

        Args:
            dice_weight (float): Weight for Dice Loss.
            lovasz_weight (float): Weight for Lovász-Softmax Loss.
            ce_weight (float): Weight for Cross Entropy Loss.
            eps (float): Epsilon to avoid division by zero in Dice Loss.
            beta (float): Beta parameter for Dice Loss.
            activation (str): Activation function ('softmax' or 'sigmoid').
            classes (str or list): Classes to include in Lovász-Softmax ('all', 'present', or list of class indices).
            per_image (bool): Whether to compute Lovász-Softmax per image.
            ignore (int, optional): Label to ignore in loss computation.
        """
        super(CombinedDiceLovaszCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLossMultiClass(eps=eps, beta=beta, activation=activation)
        self.lovasz_loss = lovasz_softmax  # Ensure this is correctly imported
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore) if ignore is not None else nn.CrossEntropyLoss()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, y_pr, y_gt):
        """
        Forward pass to compute the combined loss.

        Args:
            y_pr (torch.Tensor): Predicted logits [B, C, H, W].
            y_gt (torch.Tensor): Ground truth labels [B, H, W] or [B, C, H, W].

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Convert one-hot to class indices if necessary
        if y_gt.dim() == 4:
            y_gt = torch.argmax(y_gt, dim=1)

        # Dice Loss
        dice_loss_value = self.dice_loss(y_pr, y_gt)

        # Cross Entropy Loss
        ce_loss_value = self.ce_loss(y_pr, y_gt)

        # Lovász-Softmax Loss
        probas = F.softmax(y_pr, dim=1)
        lovasz_loss_value = self.lovasz_loss(
            probas,
            y_gt,
            classes=self.classes,
            per_image=self.per_image,
            ignore=self.ignore
        )

        # Combined Loss
        combined_loss = (
            self.dice_weight * dice_loss_value +
            self.ce_weight * ce_loss_value +
            self.lovasz_weight * lovasz_loss_value
        )

        return combined_loss




# loss = CombinedDiceLovaszLoss(
#     dice_weight=1.0,
#     lovasz_weight=1.0,
#     eps=1e-6,
#     beta=1.0,
#     activation='softmax',
#     classes='present',  # Include all classes
#     per_image=False,  # Compute loss across the entire batch
#     ignore=None  # Set to your ignore index if applicable
# )

loss = CombinedDiceLovaszCELoss(
    dice_weight=0.0,
    lovasz_weight=1.0,
    ce_weight=1.0,  
    eps=1e-6,
    beta=1.0,
    activation='softmax',
    classes='present',  # or 'present' or list of class indices
    per_image=True,
    ignore=None  # Set to your ignore index if applicable
)

# loss = DiceLossC()
# loss = nn.CrossEntropyLoss()
# loss = Focal_Loss()
# loss = DiceLossMultiClass()
loss = CombinedDiceCrossEntropyLoss()
# loss = CombinedDiceCrossEntropyEdgeLoss(dice_weight=1.0, ce_weight=1.0, edge_weight=0.5)

# loss = WeightedFocalLoss(class_weights=torch.tensor([0.1, 0.2, 0.5, 0.05, 1.0, 1.0, 0.4, 1.0, 1.0 ,1.0]), idc=idc) # imaginary wegiths!!! 
# loss = Focal_Loss(idc=idc)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params_vig = count_parameters(model_dunet)
print("Model ViG Parameter Count:", num_params_vig)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("here her ehere herhehehrehhrehr ehr", DEVICE)
train_epoch = TrainEpoch(
    model_dunet, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
 
valid_epoch = ValidEpoch(
    model_dunet, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0
 
for i in range(0, 1001):
    
    print('\nEpoch: {}'.format(i+1))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # Save the model with best iou score
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model_dunet, "graphbottleneck_wood_loss.pth")
        print('Model saved!')
# max_score = 8888
 
# for i in range(0, 1001):
    
#     print('\nEpoch: {}'.format(i+1))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
#     print("valid_logs: ", valid_logs)
#     # Save the model with best iou score
#     if max_score > valid_logs['DiceLossMultiClass']:
#         max_score = valid_logs['DiceLossMultiClass']
#         torch.save(model_dunet, "graphbottleneck_wood_loss.pth")
#         print('Model saved!')
        
    # if i == 10:
    #     optimizer.param_groups[0]['lr'] = 1e-5
    #     print('Decrease decoder learning rate to 1e-5!')


# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=None, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
) 
 
test_dataloader = DataLoader(test_dataset)
 
metrics = [
    IoU(threshold=0.5),
    Accuracy(threshold=0.5),
    Fscore(threshold=0.5),
    Recall(threshold=0.5),
    Precision(threshold=0.5),
]


Trained_model = torch.load('graphbottleneck_wood_loss.pth')
 
# Evaluate model on test set
test_epoch = ValidEpoch(
    model=Trained_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
 
logs = test_epoch.run(test_dataloader)