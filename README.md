# Electronic-Imaging-2025-paper-4492
**Image Segmentation: Inducing Graph-Based Learning**

## Abstract

This study explores the potential of graph neural networks (GNNs) to enhance semantic segmentation across diverse image modalities. We evaluate the effectiveness of a novel GNN-based U-Net architecture on three distinct datasets: PascalVOC, a standard benchmark for natural image segmentation, WoodScape, a challenging dataset of fisheye images commonly used in autonomous driving, introducing significant geometric distortions; and ISIC2016, a dataset of dermoscopic images for skin lesion segmentation. We compare our proposed UNet-GNN model against established convolutional neural networks (CNNs) based segmentation models, including U-Net and U-Net++, as well as the transformer-based SwinUNet. Unlike these methods, which primarily rely on local convolutional operations or global self-attention, GNNs explicitly model relationships between image regions by constructing and operating on a graph representation of the image features. This approach allows the model to capture long-range dependencies and complex spatial relationships, which we hypothesize will be particularly beneficial for handling geometric distortions present in fisheye imagery and capturing intricate boundaries in medical images. Our analysis demonstrates the versatility of GNNs in addressing diverse segmentation challenges and highlights their potential to improve segmentation accuracy in various applications, including autonomous driving and medical image analysis.
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
