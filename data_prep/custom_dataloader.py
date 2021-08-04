# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Classes to load KITTI and Cityscapes data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import sys
import shutil
import re
from absl import logging
import numpy as np
import imageio
from PIL import Image
import cv2
file_path = os.path.abspath(__file__)
# Root directory of the RCNN repository
MRCNN_DIR = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Mask_RCNN')
sys.path.append(MRCNN_DIR)  # To find local version of the library
sys.path.append(os.path.join(MRCNN_DIR, "samples/coco/"))
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Video(object):
  """ Make dataloader from any videos in a folder"""

  def __init__(self,
               dataset_dir,
               img_height=128,
               img_width=416,
               seq_length=3,
               sample_every=1,
               fps=10,
               gen_mask=False):
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.gen_mask = gen_mask
    self.sample_every = sample_every
    self.fps = fps
    self.video_files = self.collect_videos()
    self.videos2images()
    self.frames = self.collect_frames()
    self.num_frames = len(self.frames)
    self.num_train = self.num_frames
    logging.info('Total frames collected: %d', self.num_frames)

    if self.gen_mask:
        self._initialize_mrcnn_model()

  def _initialize_mrcnn_model(self):
    MODEL_DIR = os.path.join(MRCNN_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    config = InferenceConfig()
    # Create model object in inference mode.
    model =  modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    self.model = model

  def _compute_mask(self, image):
    results = self.model.detect([image], verbose=1)
    result = results[0]
    # Prepare black image
    mask_base = np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.uint8)
    after_mask_img = image.copy()
    color = (10, 10, 10) #white
    number_of_objects=len(result['masks'][0,0])
    mask_img=mask_base
    for j in range(0,number_of_objects):
        mask = result['masks'][:, :, j]
        mask_img = visualize.apply_mask(mask_base, mask, color,alpha=1)
    return mask_img

  def collect_videos(self):
      video_files = glob.glob(os.path.join(self.dataset_dir, '*.mp4'))
      return video_files

  def videos2images(self):
      """ Convert videos to temporary images to be processed"""

      vid2img(self.video_files, self.dataset_dir, fps=self.fps,
              target_h=self.img_height, target_w=self.img_width, )

  def delete_temp_images(self):
      for video in self.video_files:
          temp_vfolder = video.split('.mp4')[0]
          try:
              shutil.rmtree(temp_vfolder)
          except:
              assert False, f'error occurred while deleting {temp_vfolder}'

  def collect_frames(self):
    """Create a list of unique ids for available frames."""

    video_list = []
    for vdir in os.listdir(self.dataset_dir):
        if os.path.isdir(os.path.join(self.dataset_dir, vdir)):
            video_list.append(vdir)
    logging.info('video_list: %s', video_list)
    frames = []
    for video in video_list:
      im_files = glob.glob(os.path.join(self.dataset_dir, video, '*.jpg'))
      im_files = sorted(im_files, key=natural_keys)
      # Adding 3 crops of the video.
      frames.extend(['A' + video + '/' + os.path.basename(f) for f in im_files])
      frames.extend(['B' + video + '/' + os.path.basename(f) for f in im_files])
      frames.extend(['C' + video + '/' + os.path.basename(f) for f in im_files])
    return frames

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(target_index):
      return False
    example = self.load_example(target_index)
    return example

  def load_intrinsics(self, unused_frame_idx, crop_top):
    """Load intrinsics."""
    # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
    # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # # iPhone: These numbers are for images with resolution 720 x 1280.
    # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
    intrinsics = np.array([[1783.8784, 0.0, 309.5386],
                           [0, 1776.7683, 311.0264-crop_top],
                           [0, 0, 1.0]])
    return intrinsics

  def is_valid_sample(self, target_index):
    """Checks whether we can find a valid sequence around this frame."""
    target_video, _ = self.frames[target_index].split('/')
    start_index, end_index = get_seq_start_end(target_index,
                                               self.seq_length,
                                               self.sample_every)
    if start_index < 0 or end_index >= self.num_frames:
      return False
    start_video, _ = self.frames[start_index].split('/')
    end_video, _ = self.frames[end_index].split('/')
    if target_video == start_video and target_video == end_video:
      return True
    return False

  def load_image_raw(self, frame_id):
    """Reads the image and crops it according to first letter of frame_id."""
    crop_type = frame_id[0]
    img_file = os.path.join(self.dataset_dir, frame_id[1:])
    img = imageio.imread(img_file)
    allowed_height = int(img.shape[1] * self.img_height / self.img_width)
    # Starting height for the middle crop.
    mid_crop_top = int(img.shape[0] / 2 - allowed_height / 2)
    # How much to go up or down to get the other two crops.
    height_var = int(mid_crop_top / 3)
    if crop_type == 'A':
      crop_top = mid_crop_top - height_var
      cy = allowed_height / 2 + height_var
    elif crop_type == 'B':
      crop_top = mid_crop_top
      cy = allowed_height / 2
    elif crop_type == 'C':
      crop_top = mid_crop_top + height_var
      cy = allowed_height / 2 - height_var
    else:
      raise ValueError('Unknown crop_type: %s' % crop_type)
    crop_bottom = crop_top + allowed_height + 1
    return img[crop_top:crop_bottom, :, :], cy, crop_top

  def load_image_sequence(self, target_index):
    """Returns a list of images around target index."""
    start_index, end_index = get_seq_start_end(target_index,
                                               self.seq_length,
                                               self.sample_every)
    image_seq = []
    if self.gen_mask:
        mask_seq = []

    for idx in range(start_index, end_index + 1, self.sample_every):
      frame_id = self.frames[idx]
      img, cy, crop_top = self.load_image_raw(frame_id)
      if idx == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      # notice the default mode for RGB images is BICUBIC
      img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))
      if self.gen_mask:
          mask_seq.append(self._compute_mask(img))
      image_seq.append(img)
    if self.gen_mask:
        return image_seq, mask_seq, zoom_x, zoom_y, cy, crop_top
    else:
        return image_seq, zoom_x, zoom_y, cy, crop_top

  def load_example(self, target_index):
    """Returns a sequence with requested target frame."""
    example = {}
    if self.gen_mask:
        image_seq, mask_seq, zoom_x, zoom_y, cy, crop_top = self.load_image_sequence(target_index)
        example['mask_seq'] = mask_seq
    else:
        image_seq, zoom_x, zoom_y, cy, crop_top = self.load_image_sequence(target_index)
    target_video, target_filename = self.frames[target_index].split('/')
    # Put A, B, C at the end for better shuffling.
    target_video = target_video[1:] + target_video[0]
    intrinsics = self.load_intrinsics(target_index, crop_top)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_video
    example['file_name'] = target_filename.split('.')[0]
    return example

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out

def get_resource_path(relative_path):
  return relative_path

def get_seq_start_end(target_index, seq_length, sample_every=1):
  """Returns absolute seq start and end indices for a given target frame."""
  half_offset = int((seq_length - 1) / 2) * sample_every
  end_index = target_index + half_offset
  start_index = end_index - (seq_length - 1) * sample_every
  return start_index, end_index

def atoi(text):
  return int(text) if text.isdigit() else text

def natural_keys(text):
  return [atoi(c) for c in re.split(r'(\d+)', text)]

def vid2img(video_files, save_dir, fps=5, crop=True, crop_h1=0, crop_h2=720,
            crop_w1=0, crop_w2=1280, resize=False, target_h=128, target_w=416,
            shift_h=0.15, shift_w=0.0, img_ext='.jpg'):

    parent = os.path.abspath(
            os.path.expanduser(save_dir)
            )

    for video in video_files:

        print(f'converting {video} into images')

        # get the camera intrinsics once per video
        #k = get_cam_intrinsics(data_name, vf)

        vidcap = cv2.VideoCapture(video)
        
        # get the approximate number of frames
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # get the approximate frame rate 
        raw_fps = vidcap.get(cv2.CAP_PROP_FPS)

        if fps:
            # original fps
            assert raw_fps >= fps, "the specified fps is higher than the raw video"
            # the period of saving images from the video
            period = round(raw_fps/fps)
        else:
            # save every frame
            period = 1

        if frame_count==0:
            print('frame count cannot be read')
        else:
            print(f'there are {frame_count} frames in the video')

        print(f'save images at {fps} fps')

        # the folder to save images of a specific root
        path = os.path.join(
                parent,
                os.path.basename(video).split('.')[0]
                )

        if not os.path.exists(path):
            os.makedirs(path)

        count = 0
        while True:

            success, image = vidcap.read()

            # repeat if video reading has not started
            if vidcap.get(cv2.CAP_PROP_POS_MSEC) == 0.0:
                success, image = vidcap.read()

            if success:
                if count%period == 0:
                    save_idx = count//period
                    if crop:
                        image = image[
                                crop_h1:crop_h2,
                                crop_w1:crop_w2,
                                :]

                    if resize:
                       image, ratio, delta_u, delta_v = image_resize(image,
                                                                     target_h,
                                                                     target_w,
                                                                     shift_h,
                                                                     shift_w)

                    # save frame as jpeg file      
                    cv2.imwrite(
                            os.path.join(
                                path,
                                "{:010d}{}".format(save_idx, img_ext)
                                ),
                            image
                            )

                if count%500 == 0:
                    print(f'{count} raw frames have been processed')

                count+=1
            else:
                break

def image_resize(image, target_h, target_w, shift_h, shift_w,
                 inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the raw image size

    is_pil = isinstance(image, Image.Image)

    if is_pil:
        image = np.array(image)

    (raw_h, raw_w) = image.shape[:2]

    assert raw_h >= target_h, 'must be downscaling'
    assert raw_w >= target_w, 'must be downscaling'

    if target_h/raw_h <= target_w/raw_w:
        # calculate the ratio of the width and construct the dimensions
        r = target_w / float(raw_w)
        dim = (target_w, int(raw_h * r))

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]
        
        start = int(new_h*shift_h)
        end = start + target_h
       
        assert start >=0
        assert end <= new_h

        image = image[start:end,:,:]

        delta_u = 0
        delta_v = start  

    else: 
        # calculate the ratio of the height and construct the dimensions
        r = target_h / float(raw_h)
        dim = (int(raw_w * r), target_h)

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]

        start = int(new_w*shift_w)
        end = start + target_w
        image = image[:,start:end,:]

        assert start >=0
        assert end <= new_w

        image = image[:,start:end,:]

        delta_u = start
        delta_v = 0

    if is_pil:
        image = Image.fromarray(image)

    return image, r, delta_u, delta_v
