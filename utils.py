import torch
import torchvision
# from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import albumentations as albu
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch import Tensor
# from focal_loss.focal_loss import FocalLoss

import albumentations as albu




def get_training_augmentation():
    # Reduced image size to lessen GPU memory usage
    train_transform = [
        # Resize to a size divisible by 32
        albu.Resize(512, 512, p=1),  # Reduced sizes, ensure they are suitable for your model
        # albu.Resize(512, 5120, p=1),
        albu.PadIfNeeded(min_height=512, min_width=512, p=1),  # Adjust padding to make divisible by 32
        albu.HorizontalFlip(p=0.5),
        # Uncomment and adjust the following block if needed
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
        ], p=0.9),
        albu.GaussNoise(p=0.2),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation():
    # Reduced image size for validation to match the training augmentation
    test_transform = [
        albu.Resize(512, 512, p=1),
        albu.PadIfNeeded(min_height=512, min_width=512, p=1)  # Adjust padding as required
    ]
    return albu.Compose(test_transform)





# def get_training_augmentation():
#     # Enhanced augmentation pipeline for diverse transformations
#     train_transform = [
#         # Resize and padding to ensure consistent input size
#         albu.Resize(512, 512, p=1),  # Adjust size if necessary
#         albu.PadIfNeeded(
#             min_height=512, 
#             min_width=512, 
#             border_mode=0,  # Equivalent to cv2.BORDER_CONSTANT
#             value=0,  # Padding value (black)
#             p=1
#         ),
        
#         # Flipping and rotation
#         albu.HorizontalFlip(p=0.5),
#         albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=1, p=0.8),  # BORDER_REFLECT
        
#         # Color augmentations
#         albu.OneOf([
#             albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
#             albu.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1),
#             albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)
#         ], p=0.9),
        
#         # Noise and blur
#         albu.OneOf([
#             albu.GaussNoise(var_limit=(10.0, 50.0), p=1),
#             albu.MedianBlur(blur_limit=5, p=1),
#             albu.MotionBlur(blur_limit=5, p=1)
#         ], p=0.5),
        
#         # Sharpen and distortion
#         albu.OneOf([
#             albu.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, border_mode=1, p=1),  # BORDER_REFLECT
#             albu.GridDistortion(num_steps=5, distort_limit=0.03, border_mode=1, p=1),
#             albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),  # Updated to `Sharpen`
#         ], p=0.7),
        
#         # Random cropping and resizing for better scale invariance
#         albu.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
        
#         # Channel dropout for robustness to missing data
#         albu.OneOf([
#             albu.ChannelDropout(p=0.2),
#             albu.ChannelShuffle(p=0.2),
#         ], p=0.1),
        
#         # Cutout for occlusion robustness
#         # albu.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
#     ]
#     return albu.Compose(train_transform)
# def get_training_augmentation():
#     # Enhanced augmentation pipeline with fisheye-specific augmentations
#     train_transform = [
#         # Resize and padding to ensure consistent input size
#         albu.Resize(224, 224, p=1),
#         albu.PadIfNeeded(
#             min_height=224, 
#             min_width=224, 
#             border_mode=0,  # Equivalent to cv2.BORDER_CONSTANT
#             value=0,  # Padding value (black)
#             p=1
#         ),
        
#         # Flipping and rotation
#         albu.HorizontalFlip(p=0.5),
#         albu.ShiftScaleRotate(
#             shift_limit=0.05, 
#             scale_limit=0.1, 
#             rotate_limit=180,  # Allowing full rotation for fisheye data
#             border_mode=1,  # BORDER_REFLECT
#             p=0.8
#         ),
        
#         # Color augmentations
#         albu.OneOf([
#             albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
#             albu.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=1),  # Increased clip limit for more contrast adjustment
#             albu.HueSaturationValue(
#                 hue_shift_limit=30, 
#                 sat_shift_limit=40, 
#                 val_shift_limit=30, 
#                 p=1
#             )
#         ], p=0.9),
        
#         # Fisheye-specific distortions
#         albu.OneOf([
#             albu.OpticalDistortion(
#                 distort_limit=0.1, 
#                 shift_limit=0.1, 
#                 border_mode=0,  # BORDER_CONSTANT
#                 p=1
#             ),
#             albu.GridDistortion(
#                 num_steps=5, 
#                 distort_limit=0.05, 
#                 border_mode=0, 
#                 p=1
#             ),
#         ], p=0.7),
        
#         # Noise and blur
#         albu.OneOf([
#             albu.GaussNoise(var_limit=(10.0, 50.0), p=1),
#             albu.MedianBlur(blur_limit=5, p=1),
#             albu.MotionBlur(blur_limit=5, p=1)
#         ], p=0.5),
        
#         # Sharpening
#         albu.Sharpen(
#             alpha=(0.3, 0.7),  # Stronger sharpening range
#             lightness=(0.6, 1.0), 
#             p=0.7
#         ),
        
#         # Random cropping and resizing for scale invariance
#         albu.RandomResizedCrop(
#             height=224, 
#             width=224, 
#             scale=(0.6, 1.0),  # Allow more aggressive cropping
#             ratio=(0.8, 1.2),  # Increased ratio variation
#             p=0.9
#         ),
        
#         # Channel dropout for robustness to missing data
#         albu.OneOf([
#             albu.ChannelDropout(p=0.3),
#             albu.ChannelShuffle(p=0.3),
#         ], p=0.2),
        
#         # Cutout for occlusion robustness
#         # albu.Cutout(
#         #     num_holes=4, 
#         #     max_h_size=48, 
#         #     max_w_size=48, 
#         #     fill_value=0,  # Black fill
#         #     p=0.4
#         # ),
#     ]
#     return albu.Compose(train_transform)





def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

# def get_training_augmentation():
#     train_transform = [
 
#         albu.Resize(256, 416, p=1),
#         albu.PadIfNeeded(256, 416),
#         albu.HorizontalFlip(p=0.5),
 
#         albu.OneOf([
#             albu.RandomBrightnessContrast(
#                   brightness_limit=0.4, contrast_limit=0.4, p=1),
#             albu.CLAHE(p=1),
#             albu.HueSaturationValue(p=1)
#             ],
#             p=0.9,
#         ),
 
#         # albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.GaussNoise(p=0.2),
#     ]
#     return albu.Compose(train_transform)


# def get_validation_augmentation():
#     """Add paddings to make image shape divisible by 32"""
#     test_transform = [
#         albu.PadIfNeeded(256, 416)
#     ]
#     return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
 
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
        # albu.Resize(512, 512),
    ]
    return albu.Compose(_transform)








def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()