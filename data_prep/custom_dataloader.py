# Copyright All Rights Reserved.
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

"""Classes to load your own videos"""
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
               cut = False,
               crop = 'multi',
               shift_h = 0.0,
               fps=10,
               img_ext='png',
               gen_mask=False):

    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.gen_mask = gen_mask
    self.sample_every = sample_every
    self.cut = cut
    self.crop = crop
    self.shift_h = shift_h
    self.fps = fps
    self.img_ext = img_ext
    # TODO: check if converted images exist
    self.videos = self.collect_videos()
    self.vid2img()
    self.video_dirs = self.collect_video_dirs()
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
      """ Collect absolute paths of the video files for conversion"""

      videos = glob.glob(os.path.join(self.dataset_dir, '*.mp4'))
      return videos

  def collect_video_dirs(self):
      """ Return a list of names of all the video directories"""

      # names of all the video directories
      video_dirs = []

      # iterate through all the file names
      for item in os.listdir(self.dataset_dir):
          # collect directories named with the videos' names
          if os.path.isdir(os.path.join(self.dataset_dir, item)):
              # only collect folder names
              video_dirs.append(item)

      return video_dirs

  def vid2img(self):
      """ Convert videos to images without rescaling """

      fps = self.fps
      save_dir = self.dataset_dir
      img_ext = self.img_ext

      for video in self.videos:
          print(f'converting {video} into images')
          vidcap = cv2.VideoCapture(video)
            
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

            # the folder to save images of a specific root
          path = os.path.join(
                  save_dir,
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

                      if self.cut:
                          image = self._initial_crop(image)
                      cv2.imwrite(
                              os.path.join(
                                  path, "{:010d}.{}".format(save_idx, img_ext)
                                  ),
                              image
                              )

                  if count%500 == 0:
                      print(f'{count} raw frames have been processed')

                  count+=1
              else:
                  break

  def _initial_crop(self, img, crop_h=720, crop_w=1280):
      """ Crop out H720 x W1280 of a image

      This Function is to crop out a portion out of the input frame. 
      Since there is not following adjustment of intrinsics, it should only be 
      applied when the frame is composed of concatenation of images from 
      different camera.
      """

      return img[:crop_h, :crop_w, :]

  def collect_frames(self):
    """Create a list of unique ids for available frames."""

    frames = []
    for video_dir in self.video_dirs:
      # absolute paths
      im_files = glob.glob(os.path.join(self.dataset_dir, video_dir, f'*.{self.img_ext}'))
      # sort images in a video directory; this sorting works even 
      # when the image indices are not formated to same digits
      im_files = sorted(im_files, key=natural_keys)

      if self.crop == 'multi':
          # Adding 3 crops of the video.
          frames.extend(['A' + video_dir + '/' + os.path.basename(f) for f in im_files])
          frames.extend(['B' + video_dir + '/' + os.path.basename(f) for f in im_files])
          frames.extend(['C' + video_dir + '/' + os.path.basename(f) for f in im_files])
      elif self.crop == 'single':
          frames.extend(['S' + video_dir + '/' + os.path.basename(f) for f in im_files])
      else:
          raise NotImplementedError(f'crop {self.crop} not supported')
      
    return frames

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(target_index):
      return False
    example = self.load_example(target_index)
    return example


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

    # image shape (H, W, C)
    # assume H would be crop
    allowed_height = int(img.shape[1] * self.img_height / self.img_width)

    # Starting height for the middle crop.
    mid_crop_top = int(img.shape[0] / 2 - allowed_height / 2)
    # How much to go up or down to get the other two crops.
    # hard coded as one-third of the top cropping
    # crop_top will be used to adjust the princial point y in the intrinsics
    # due to the cropping
    height_var = int(mid_crop_top / 3)
    if crop_type == 'A':
      crop_top = mid_crop_top - height_var
    elif crop_type == 'B':
      crop_top = mid_crop_top
    elif crop_type == 'C':
      crop_top = mid_crop_top + height_var
    elif crop_type == 'S':
      crop_top = int(self.shift_h *  img.shape[0])
    else:
      raise ValueError('Unknown crop_type: %s' % crop_type)

    crop_bottom = crop_top + allowed_height + 1

    return img[crop_top:crop_bottom, :, :], crop_top

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
      img, crop_top = self.load_image_raw(frame_id)

      # since images have been cropped according to the
      # aspect ratio specified by (img_heigh, img_width)
      # zoom_y and zoom_x should be very close
      if idx == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]

      # notice the default mode for RGB images rescaling is BICUBIC
      img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))

      if self.gen_mask:
          mask_seq.append(self._compute_mask(img))

      image_seq.append(img)

    if self.gen_mask:
        return image_seq, mask_seq, zoom_x, zoom_y, crop_top
    else:
        return image_seq, zoom_x, zoom_y, crop_top

  def load_example(self, target_index):
    """Returns a sequence with requested target frame."""

    example = {}

    if self.gen_mask:
        image_seq, mask_seq, zoom_x, zoom_y, crop_top = self.load_image_sequence(target_index)
        example['mask_seq'] = mask_seq
    else:
        image_seq, zoom_x, zoom_y, crop_top = self.load_image_sequence(target_index)

    target_video, target_filename = self.frames[target_index].split('/')
    if self.crop == 'multi':
        # Put A, B, C at the end for better shuffling.
        target_video = target_video[1:] + target_video[0]
    elif self.crop == 'single':
        target_video = target_video[1:]

    # first adjust intrinsics due to cropping
    intrinsics = self.load_intrinsics(target_index, crop_top)
    # then adjust intrinsics due to rescaling
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_video
    example['file_name'] = target_filename.split('.')[0]
    return example

  def load_intrinsics(self, unused_frame_idx, crop_top):
    """Load intrinsics."""
    intrinsics = np.array([[1783.8784, 0.0, 309.5386],
                           [0, 1776.7683, 311.0264-crop_top],
                           [0, 0, 1.0]])
    return intrinsics

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out

  def delete_temp_images(self):
      """ Delete the intially converted images from videos """
      for video_dir in self.video_dirs:
          try:
              shutil.rmtree(video_dir)
          except:
              assert False, f'error occurred while deleting {temp_vfolder}'

def get_seq_start_end(target_index, seq_length, sample_every=1):
  """Returns absolute seq start and end indices for a given target frame."""
  half_offset = int((seq_length - 1) / 2) * sample_every
  end_index = target_index + half_offset
  start_index = end_index - (seq_length - 1) * sample_every
  return start_index, end_index

def atoi(text):
  """ Transform a string to digit if possible """
  return int(text) if text.isdigit() else text

def natural_keys(text):
  return [atoi(c) for c in re.split(r'(\d+)', text)]

