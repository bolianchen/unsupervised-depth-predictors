# Unsupervised Depth Predictors
## Introduction
We are developing an algorithm prototype to estimate depth for advanced driver-assistance systems (ADAS). A branch of unsupervised deep learning methods pioneered by SfMLearner seems promising due to the lack of lidars or stereo cameras on our systems. Eventually, we also like to tackle the artifacts caused by relative motions of objects to the camera. For the purpose, this repository aims to facilitate the use of several Google's publications by revising and reorganizing their official codes.

<details><summary><strong>struct2depth</strong></summary>
<p>

- paper: [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152)
- codes: [tensorflow/models, commit 36101ab](https://github.com/tensorflow/models/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9)
    
</p>
</details>

<details><summary><strong>vid2depth</strong></summary>
<p>

- paper: [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522)
- codes: [tensorflow/models, commit 37ec178](https://github.com/tensorflow/models/tree/37ec31714f68255532b4c35f117bc33fd7f90692)
    
</p>
</details>

<details><summary><strong>depth_from_video_in_the_wild</strong></summary>
<p>

- paper: [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)
- codes: [google-research/google-research, commit 57b6017](https://github.com/google-research/google-research/tree/57b60e7a7a5efc358adf4041a062ae435e6155be)

</p>
</details>
    
<details><summary><strong>depth_and_motion_learning</strong></summary>
<p>

- paper: [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/abs/2010.16404)
- codes: [google-research/google-research, commit 57b6017](https://github.com/google-research/google-research/tree/57b60e7a7a5efc358adf4041a062ae435e6155be)

</p>
</details>

Segmentation masks of the training images are generated based on [**matterport/Mask_RCNN**](https://github.com/matterport/Mask_RCNN/tree/3deaec5d902d16e1daf56b62d5971d428dc920bc). 

## Environment Setup

The codes are tested with tensorflow 1.15 and its official binaries are built with cuda10.0 that might be inconsistent with yours. Using conda environments is likely to prevent you from suffering.
    
```
conda create -n your_virtual_env_name python=3.6
conda install -c anaconda cudatoolkit=10.0 cudnn=7.6.5
pip install -r requirements.txt
```

## Data Preparation

<details><summary><strong>Download ResNet-18 Checkpoint</strong></summary>
<p>

I created a script to download tensorflow resnet-18 checkpoint trained on ImageNet by referring to [**dalgu90/resnet-18-tensorflow**](https://github.com/dalgu90/resnet-18-tensorflow/tree/49eb67c3c57258537c0dcbab5cdf2c38bb1af776).
    
```
./imagenet_ckpt_downloader.sh
```
After runing the script, the checkpoint will be saved to `Imagenet_ckpt` folder within the project folder.
```
Imagenet_ckpt/
├── checkpoint
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
└── model.ckpt.meta
```
    
</p>
</details>

<details><summary><strong>KITTI Data</strong></summary>
<p>

#### Download Raw Data

```bash
$ version=full_version  # choose among (tiny_version, mini_version, full_version)
$ ./kitti_raw_downloader.sh $version
```

```
KITTI_raw/
└── 2011_09_26
    ├── 2011_09_26_drive_0001_sync
    │   ├── image_00
    │   ├── image_01
    │   ├── image_02
    │   ├── image_03
    │   ├── oxts
    │   └── velodyne_points
    ├── calib_cam_to_cam.txt
    ├── calib_imu_to_velo.txt
    └── calib_velo_to_cam.txt
```
    
</p>
</details>

<details><summary><strong>Your Own Videos</strong></summary>
<p>

    
</p>
</details>



## Model Training
#### struct2depth:

```
$python struct2depth/train.py --logtostderr \
                              --checkpoint_dir ../test_struct2depth \
                              --data_dir ./KITTI_processed \
                              --architecture resnet \
                              --imagenet_ckpt ./Imagenet_ckpt/model.ckpt
                              --epochs 20
```
#### vid2depth:
```
$
```
- depth_from_video_in_the_wild:
```
$ python -m depth_from_video_in_the_wild.train --checkpoint_dir=$MY_CHECKPOINT_DIR \
                                               --data_dir=$MY_DATA_DIR \
                                               --imagenet_ckpt=$MY_IMAGENET_CHECKPOINT
```

#### depth_and_motion_learning:
```
$ python -m depth_and_motion_learning.depth_motion_field_train --model_dir=../test_motion \
                                                               --param_overrides='{
                                                                 "model": { 
                                                                   "input": {
                                                                     "data_path": "KITTI_processed/train.txt"
                                                                   }
                                                                 },
                                                                 "trainer": {
                                                                   "init_ckpt": "Imagenet_ckpt/model.ckpt",
                                                                   "init_ckpt_type": "imagenet"
                                                                 }
                                                               }'
```

## Inference
#### struct2depth:
```

```
