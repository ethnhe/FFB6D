# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) on Semantic KITTI.

## Preparation

1. Clone this repository

2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.

3. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).

```
sh compile_op.sh
```
4. Download the [Semantic KITTI dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:
  ```
  python utils/data_prepare_semantickitti.py
  ```
   Note: Please change the dataset path in the 'data_prepare_semantickitti.py' with your own path.


## Train a model

  ```
  python main_SemanticKITTI.py
  ```

This repository implements the official version as much as possible. There are following differences:

1) We use Conv2D instead of ConvTranspose2D in the decoder part. Since the strides are [1,1] and conv kernels are [1,1], we think the Conv2D and ConvTranspose2D are the same in this condition.

2) We delate the bias in the Conv2D which is followed by a BN layer, since the bias is meaningless in the Conv2D if the Conv2D is followed by a BN layer.

3) We evaluate the network for one epoch after every 10 training epoches.


## Results
We train this network for 60 epoches, and the eval results in the 8-th Sequence are as follows:

```
mean IoU: 52.6

IoU: 
car:           93.07
bicycle:       16.70
motorcycle:    20.64
truck:         67.74
other-vehicle: 41.46
person:        48.79
bicyclist:     65.74
motorcyclist:   0.00
road:          91.44
parking:       42.21
sidewalk:      76.50
other-ground:   6.15
building:      85.55
fence:         38.86
vegetation:    84.19
trunk:         58.49
terrain:       73.63
pole:          51.63
traffic-sign:  36.05
```

There are some differences with the results in the paper, but we think the overall results is acceptable. 

The checkpoint is in the output folder.
