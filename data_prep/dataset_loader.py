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

CITYSCAPES_CROP_BOTTOM = True  # Crop bottom 25% to remove the car hood.
CITYSCAPES_CROP_PCT = 0.75
CITYSCAPES_SAMPLE_EVERY = 2  # Sample every 2 frames to match KITTI frame rate.
BIKE_SAMPLE_EVERY = 6  # 5fps, since the bike's motion is slower.

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class KittiRaw(object):
  """Reads KITTI raw data files."""

  def __init__(self,
               dataset_dir,
               split,
               load_pose=False,
               img_height=128,
               img_width=416,
               seq_length=3,
               gen_mask=False):
    static_frames_file = 'data_prep/kitti/static_frames.txt'
    test_scene_file = 'data_prep/kitti/test_scenes_' + split + '.txt'
    with open(get_resource_path(test_scene_file), 'r') as f:
      test_scenes = f.readlines()
    self.test_scenes = [t[:-1] for t in test_scenes]
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.gen_mask = gen_mask
    self.load_pose = load_pose
    self.cam_ids = ['02', '03']
    self.date_list = [
        '2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03'
    ]
    self.collect_static_frames(static_frames_file)
    self.collect_train_frames()
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

  def collect_static_frames(self, static_frames_file):
    with open(get_resource_path(static_frames_file), 'r') as f:
      frames = f.readlines()
    self.static_frames = []
    for fr in frames:
      if fr == '\n':
        continue
      unused_date, drive, frame_id = fr.split(' ')
      fid = '%.10d' % (np.int(frame_id[:-1]))
      for cam_id in self.cam_ids:
        self.static_frames.append(drive + ' ' + cam_id + ' ' + fid)

  def collect_train_frames(self):
    """Creates a list of training frames."""
    all_frames = []
    for date in self.date_list:
      date_dir = os.path.join(self.dataset_dir, date)
      if os.path.isdir(date_dir):
          drive_set = os.listdir(date_dir)
          for dr in drive_set:
            drive_dir = os.path.join(date_dir, dr)
            if os.path.isdir(drive_dir):
              if dr[:-5] in self.test_scenes:
                continue
              for cam in self.cam_ids:
                img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
                num_frames = len(glob.glob(img_dir + '/*[0-9].png'))
                for i in range(num_frames):
                  frame_id = '%.10d' % i
                  all_frames.append(dr + ' ' + cam + ' ' + frame_id)

    assert len(all_frames)>0, 'no kitti data found in the dataset_dir'

    for s in self.static_frames:
      try:
        all_frames.remove(s)
      except ValueError:
        pass

    assert len(all_frames)>0, 'all data are static_frames'

    self.train_frames = all_frames
    self.num_train = len(self.train_frames)

  def is_valid_sample(self, frames, target_index):
    """Checks whether we can find a valid sequence around this frame."""
    num_frames = len(frames)
    target_drive, cam_id, _ = frames[target_index].split(' ')
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    # check if the indices of the start and end are out of the range
    if start_index < 0 or end_index >= num_frames:
      return False
    start_drive, start_cam_id, _ = frames[start_index].split(' ')
    end_drive, end_cam_id, _ = frames[end_index].split(' ')
    # check if the scenes and cam_ids are the same 
    if (target_drive == start_drive and target_drive == end_drive and
        cam_id == start_cam_id and cam_id == end_cam_id):
      return True
    return False

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(self.train_frames, target_index):
      return False
    example = self.load_example(self.train_frames, target_index)
    return example

  def load_image_sequence(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    image_seq = []
    if self.gen_mask:
        mask_seq = []
    for index in range(start_index, end_index + 1):
      drive, cam_id, frame_id = frames[index].split(' ')
      img = self.load_image_raw(drive, cam_id, frame_id)
      if index == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      # notice the default mode for RGB images is BICUBIC
      img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))
      if self.gen_mask:
          mask_seq.append(self._compute_mask(img))
      image_seq.append(img)
    if self.gen_mask:
        return image_seq, mask_seq, zoom_x, zoom_y
    else:
        return image_seq, zoom_x, zoom_y

  def load_pose_sequence(self, frames, target_index):
    """Returns a sequence of pose vectors for frames around the target frame."""
    target_drive, _, target_frame_id = frames[target_index].split(' ')
    target_pose = self.load_pose_raw(target_drive, target_frame_id)
    start_index, end_index = get_seq_start_end(target_frame_id, self.seq_length)
    pose_seq = []
    for index in range(start_index, end_index + 1):
      if index == target_frame_id:
        continue
      drive, _, frame_id = frames[index].split(' ')
      pose = self.load_pose_raw(drive, frame_id)
      # From target to index.
      pose = np.dot(np.linalg.inv(pose), target_pose)
      pose_seq.append(pose)
    return pose_seq

  def load_example(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    example = {}
    if self.gen_mask:
        image_seq, mask_seq, zoom_x, zoom_y = self.load_image_sequence(frames, target_index)
        example['mask_seq'] = mask_seq
    else:
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, target_index)
    target_drive, target_cam_id, target_frame_id = (
        frames[target_index].split(' '))
    intrinsics = self.load_intrinsics_raw(target_drive, target_cam_id)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_drive + '_' + target_cam_id + '/'
    example['file_name'] = target_frame_id
    if self.load_pose:
      pose_seq = self.load_pose_sequence(frames, target_index)
      example['pose_seq'] = pose_seq
    return example

  def load_pose_raw(self, drive, frame_id):
    date = drive[:10]
    pose_file = os.path.join(self.dataset_dir, date, drive, 'poses',
                             frame_id + '.txt')
    with open(pose_file, 'r') as f:
      pose = f.readline()
    pose = np.array(pose.split(' ')).astype(np.float32).reshape(3, 4)
    pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape((1, 4))))
    return pose

  def load_image_raw(self, drive, cam_id, frame_id):
    date = drive[:10]
    img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cam_id,
                            'data', frame_id + '.png')
    img = imageio.imread(img_file)
    return img

  def load_intrinsics_raw(self, drive, cam_id):
    date = drive[:10]
    calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')
    filedata = self.read_raw_calib_file(calib_file)
    p_rect = np.reshape(filedata['P_rect_' + cam_id], (3, 4))
    intrinsics = p_rect[:3, :3]
    return intrinsics

  # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
  def read_raw_calib_file(self, filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
      for line in f:
        key, value = line.split(':', 1)
        # The only non-float values in these files are dates, which we don't
        # care about.
        try:
          data[key] = np.array([float(x) for x in value.split()])
        except ValueError:
          pass
    return data

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out

class Bike(object):
  """Load bike video frames."""

  def __init__(self,
               dataset_dir,
               img_height=128,
               img_width=416,
               seq_length=3,
               sample_every=BIKE_SAMPLE_EVERY):
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.sample_every = sample_every
    self.frames = self.collect_frames()
    self.num_frames = len(self.frames)
    self.num_train = self.num_frames
    logging.info('Total frames collected: %d', self.num_frames)

  def collect_frames(self):
    """Create a list of unique ids for available frames."""
    video_list = os.listdir(self.dataset_dir)
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

  def load_intrinsics(self, unused_frame_idx, cy):
    """Load intrinsics."""
    # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
    # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # # iPhone: These numbers are for images with resolution 720 x 1280.
    # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
    intrinsics = np.array([[1344.8, 0, 1280 // 2],
                           [0, 1344.8, cy],
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
    # h x w x c
    img = imageio.imread(img_file)
    # assume height cropping
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
    return img[crop_top:crop_bottom, :, :], cy

  def load_image_sequence(self, target_index):
    """Returns a list of images around target index."""
    start_index, end_index = get_seq_start_end(target_index,
                                               self.seq_length,
                                               self.sample_every)
    image_seq = []
    for idx in range(start_index, end_index + 1, self.sample_every):
      frame_id = self.frames[idx]
      img, cy = self.load_image_raw(frame_id)
      if idx == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      # notice the default mode for RGB images is BICUBIC
      img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y, cy

  def load_example(self, target_index):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y, cy = self.load_image_sequence(target_index)
    target_video, target_filename = self.frames[target_index].split('/')
    # Put A, B, C at the end for better shuffling.
    target_video = target_video[1:] + target_video[0]
    intrinsics = self.load_intrinsics(target_index, cy)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
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




class Video(object):
  """Load bike video frames."""

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
      vid2img(self.video_files, self.dataset_dir, fps=self.fps,
              target_h=self.img_height, target_w=self.img_width, )

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
