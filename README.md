# Unsupervised Depth Predictors
## Introduction
We are developing an algorithm prototype to estimate depth for advanced driver-assistance systems (ADAS). For the purpose, we include several Google's publications in this repository, which follows SfMLearner but involving innovative elements to deal with relative motions. Their official codes are revised and reorganized to facilitate the usage.

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

Codes to generate possibly mobile masks are based on [**matterport/Mask_RCNN**](https://github.com/matterport/Mask_RCNN/tree/3deaec5d902d16e1daf56b62d5971d428dc920bc). 

## Environment Setup

The codes are tested with tensorflow 1.15 and its official binaries are built with cuda10.0 that might be inconsistent with yours. Using conda environments would prevent the sufferings.
    
```
$ conda create -n your_virtual_env_name python=3.6
$ conda install -c anaconda cudatoolkit=10.0 cudnn=7.6.5
$ pip install -r requirements.txt
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

### Download Raw Data

Three versions of KITTI data, `tiny_version`, `mini_version` and `full_version`, could be chosen by passing an argument to the downloading script. The former two, respectively, consist of a single data split from one date and from all of the five dates.
```bash
$ version=full_version  # choose among (tiny_version, mini_version, full_version)
$ ./kitti_raw_downloader.sh $version
```
The downloaded files will be automatically decompressed to `KITTI_raw` folder. The structure of `mini_version` is shown as follows: 
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

### Convert Raw to Training Data

```
$ dataset_name=kitti_raw_eigen        # or kitti_raw_stereo, it only affects the contents of the test dataset
$ dataset_dir=./KITTI_raw             # folder of the KITTI raw data
$ save_dir=./KITTI_processed          # folder to save the processed data, you can choose any folder
$ python data_prep/gen_data.py --dataset_name=$dataset_name  \
                               --dataset_dir=$dataset_dir \ 
                               --save_dir=$save_dir \
                               --gen_mak       # optional, whether or not to generate possibly mobile masks
```

```
KITTI_processed/
├── 2011_09_26_drive_0001_sync_02
├── 2011_09_26_drive_0001_sync_03
├── train.txt
└── val.txt
```
```
KITTI_processed/2011_09_26_drive_0001_sync_02
├── 0000000001-fseg.png
├── 0000000001.png
├── 0000000001_cam.txt
├── 0000000002-fseg.png
├── 0000000002.png
├── 0000000002_cam.txt
.
.
.
├── 0000000106-fseg.png
├── 0000000106.png
└── 0000000106_cam.txt
```
    
</p>
</details>

<details><summary><strong>Your Own Videos</strong></summary>
<p>

```
My_Videos/
├── video_1.mp4
├── video_2.mp4
└── video_3.mp4
```



```
$ dataset_name=video
$ dataset_dir=./My_Videos            # folder of the KITTI raw data
$ save_dir=./My_Videos_processed     # folder to save the processed data, you can choose any folder.
$ crop=single                        # or multi, to determine how to crop images before rescaling them
$ python data_prep/gen_data.py --dataset_name=$dataset_name  \
                               --dataset_dir=$dataset_dir \ 
                               --save_dir=$save_dir \
                               --gen_mask       # optional, whether or not to generate possibly mobile masks
```
when `$ crop=single`
```
My_Videos_processed/
├── train.txt
├── val.txt
├── video_1
├── video_2
└── video_3
```
when `$ crop=multi`
```
My_Videos_processed/
├── train.txt
├── val.txt
├── video_1A
├── video_1B
├── video_1C
├── video_2A
├── video_2B
├── video_2C
├── video_3A
├── video_3B
└── video_3C
```

</p>
</details>



## Training
    
<details><summary><strong>struct2depth</strong></summary>
<p>

```
$ DATA_DIR=KITTI_processed # the directory to your processed data
$ MY_IMAGENET_CHECKPOINT=Imagenet_ckpt/model.ckpt
$ python struct2depth/train.py --logtostderr \
                               --checkpoint_dir $Where_To_Save_Model \
                               --data_dir $DATA_DIR \
                               --architecture resnet \
                               --imagenet_ckpt $MY_IMAGENET_CHECKPOINT
                               --epochs 20
```

</p>
</details>

<details><summary><strong>vid2depth (pending)</strong></summary>
<p>
I failed to compile the ICP op, so it's pending and not prioritized for now.
</p>
</details>

<details><summary><strong>depth_from_video_in_the_wild</strong></summary>
<p>

```
$ DATA_DIR=KITTI_processed                  # the directory to your processed data
$ MY_IMAGENET_CHECKPOINT=Imagenet_ckpt/model.ckpt
$ python -m depth_from_video_in_the_wild.train --checkpoint_dir=$WHERE_TO_SAVE_MODEL \
                                               --data_dir=$DATA_DIR               \
                                               --imagenet_ckpt=$MY_IMAGENET_CHECKPOINT
```
    

</p>
</details>

<details><summary><strong>depth_and_motion_learning</strong></summary>
<p>
    
```
$ python -m depth_and_motion_learning.depth_motion_field_train --model_dir=$WHERE_TO_SAVE_MODEL \
                                                               --epoch=20
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

</p>
</details>

## Inference

<details><summary><strong>depth_from_video_in_the_wild</strong></summary>
<p>

```
$ python -m depth_from_video_in_the_wild.depth_inference --test_file_dir=$TEST_IMAGES_DIR
                                                         --checkpoint_dir=$MODEL_CHECKPOINT \
                                                         --output_dir=$WHERE_TO_SAVE_RESULTS
                                                         --output_img_disp # output concatnation of original images with depths   
```
    
</p>
</details>

<details><summary><strong>depth_and_motion_learning</strong></summary>
<p>
    
```
$ python -m depth_and_motion_learning.depth_inference --test_file_dir=$TEST_IMAGES_DIR
                                                      --checkpoint_dir=$MODEL_CHECKPOINT \
                                                      --output_dir=$WHERE_TO_SAVE_RESULTS
                                                      --output_img_disp # output concatnation of original images with depths
```

</p>
</details>

## How to Contribute (under development)

Welcome to leave messages to the Issues panel for further discussion.
