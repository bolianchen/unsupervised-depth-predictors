# private-unsupervised-depth-predictor

This repository is to tweak the official codes of several unsupervised deep learning depth predictiors and to make them easily applied for private projects:
1. **struc2depth**:  
    - paper: [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152)
    - codes were cloned from [tensorflow/models, commit 36101ab](https://github.com/tensorflow/models/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9)
2. **vid2depth**:
    - paper: [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522)
    - codes were cloned from [tensorflow/models, commit 37ec178](https://github.com/tensorflow/models/tree/37ec31714f68255532b4c35f117bc33fd7f90692)
3. **depth_from_video_in_the_wild**:
    - paper: [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)
    - codes were cloned from the master branch of [google-research/google-research, commit 57b6017](https://github.com/google-research/google-research/tree/57b60e7a7a5efc358adf4041a062ae435e6155be)
4. **depth_and_motion_learning**: 
    - paper: [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/abs/2010.16404)
    - codes were cloned from the master branch of [google-research/google-research, commit 57b6017](https://github.com/google-research/google-research/tree/57b60e7a7a5efc358adf4041a062ae435e6155be)

The segmentation masks of training images for learning object motions are generated based on [**matterport/Mask_RCNN**](https://github.com/matterport/Mask_RCNN/tree/3deaec5d902d16e1daf56b62d5971d428dc920bc); the Tensorflow ResNet-18 checkpoint trained on ImageNet is generated based on [**dalgu90/resnet-18-tensorflow**](https://github.com/dalgu90/resnet-18-tensorflow/tree/49eb67c3c57258537c0dcbab5cdf2c38bb1af776)

## Data Preparation
### KITTI 

## Model Training
### struct2depth:
```

```

## Inference
