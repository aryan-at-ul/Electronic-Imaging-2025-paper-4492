# Electronic-Imaging-2025-paper-4492
**Image Segmentation: Inducing Graph-Based Learning**

## Abstract

This study explores how Graph Neural Networks (GNNs) can improve semantic segmentation across different types of images. We test their effectiveness on two datasets: the well-known Pascal VOC Segmentation dataset and the WoodScape dataset, which features challenging fisheye images often used in autonomous driving. Unlike typical CNNs such as U-Net, U-Net++, and the transformer-based SwinUNet—which we use for comparison—GNNs exploit pixel-to-pixel relationships to potentially identify object boundaries more accurately. By applying GNNs to both standard images and those with fisheye distortion, we can assess their ability to handle typical segmentation tasks and adapt to the unique geometry of fisheye lenses. This analysis underscores the flexibility of GNNs in managing challenging imaging scenarios and emphasizes their potential to enhance semantic segmentation accuracy in various applications, including self-driving technologies.

## Getting Started

### Datasets

To replicate the results from the paper, the following datasets are required:

- **WoodScape**  
  Refer to the instructions in the official repository:  
  [github.com/valeoai/WoodScape](https://github.com/valeoai/WoodScape)

- **Pascal VOC**  
  This dataset is already supported by the `torchvision` library.  
  *(No additional download steps are necessary if you have `torchvision`.)*

- **ISIC 2016**  
  Dermoscopic images. Download from:  
  [Kaggle: ISIC Segmentation 2016](https://www.kaggle.com/datasets/ratneshkumartiwari53/isic-segmentation-2016)

### Training Scripts

Run the training scripts with either passed or hardcoded configs. A separate training script is provided for each dataset. For example:

```bash
python train.py
