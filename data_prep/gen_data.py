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

"""Generates data for training/validation and save it to disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import multiprocessing
import os
from absl import app
from absl import flags
from absl import logging
import dataset_loader
import custom_dataloader
import numpy as np
import scipy.misc
import imageio
import tensorflow as tf

gfile = tf.gfile
FLAGS = flags.FLAGS

DATASETS = ['video', 'kitti_raw_eigen', 'kitti_raw_stereo']

flags.DEFINE_enum('dataset_name', None, DATASETS, 'Dataset name.')
flags.DEFINE_string('dataset_dir', None, 'Location for dataset source files.')
flags.DEFINE_string('save_dir', None, 'Where to save the generated data.')
flags.DEFINE_string('save_img_ext', 'png', 'image format to save')

# Note: Training time grows linearly with sequence length.  Use 2 or 3.
flags.DEFINE_integer('seq_length', 3, 'Length of each training sequence.')
flags.DEFINE_integer('img_height', 128, 'Image height.')
flags.DEFINE_integer('img_width', 416, 'Image width.')
# For the video dataset only
flags.DEFINE_bool(
        'cut', False, 'if necessary to cut h720 x w1280 '
        'during the video to image conversion')
CROPPING = ['multi_crops', 'shift_h']
flags.DEFINE_enum('crop', None, CROPPING,'how to crop the images')
flags.DEFINE_float('shift_h', 0.1, 'the ratio representing the starting '
                     'point from the top to crop img_height')
# TODO add demo to interactive adjust shift_h
flags.DEFINE_integer('fps', 10, 'frames per second to sample from videos.')
flags.DEFINE_bool( 'delete_temp', False, 'delete temporary converted images '
                   'from videos')

flags.DEFINE_integer(
    'num_threads', None, 'Number of worker threads. '
    'Defaults to number of CPU cores.')
flags.DEFINE_bool('gen_mask', False, 'Where to save the generated data.')
flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('save_dir')
flags.mark_flag_as_required('crop')

# Process data in chunks for reporting progress.
NUM_CHUNKS = 100

def _generate_data():
  """Extract sequences from dataset_dir and store them in save_dir."""
  if not gfile.Exists(FLAGS.save_dir):
    gfile.MakeDirs(FLAGS.save_dir)

  global dataloader  # pylint: disable=global-variable-undefined
  if FLAGS.dataset_name == 'video':
    dataloader = custom_dataloader.Video(FLAGS.dataset_dir,
                                         img_height=FLAGS.img_height,
                                         img_width=FLAGS.img_width,
                                         seq_length=FLAGS.seq_length,
                                         cut = FLAGS.cut,
                                         crop = FLAGS.crop,
                                         shift_h = FLAGS.shift_h,
                                         fps=FLAGS.fps,
                                         img_ext = FLAGS.save_img_ext,
                                         gen_mask=FLAGS.gen_mask)

  elif FLAGS.dataset_name == 'kitti_raw_eigen':
    dataloader = dataset_loader.KittiRaw(FLAGS.dataset_dir,
                                         split='eigen',
                                         img_height=FLAGS.img_height,
                                         img_width=FLAGS.img_width,
                                         seq_length=FLAGS.seq_length,
                                         gen_mask=FLAGS.gen_mask)
  elif FLAGS.dataset_name == 'kitti_raw_stereo':
    dataloader = dataset_loader.KittiRaw(FLAGS.dataset_dir,
                                         split='stereo',
                                         img_height=FLAGS.img_height,
                                         img_width=FLAGS.img_width,
                                         seq_length=FLAGS.seq_length,
                                         gen_mask=FLAGS.gen_mask)
  else:
    raise ValueError('Unknown dataset')

  # The default loop below uses multiprocessing, which can make it difficult
  # to locate source of errors in data loader classes.
  # Uncomment this loop for easier debugging:

  # all_examples = {}
  # for i in range(dataloader.num_train):
  #   _gen_example(i, all_examples)
  #   logging.info('Generated: %d', len(all_examples))

  all_frames = range(dataloader.num_train)
  # Split into training/validation sets. Fixed seed for repeatability.
  np.random.seed(8964)

  if FLAGS.gen_mask:
      all_examples ={}

      if not gfile.Exists(FLAGS.save_dir):
        gfile.MakeDirs(FLAGS.save_dir)

      with gfile.Open(os.path.join(FLAGS.save_dir, 'train.txt'), 'w') as train_f:
        with gfile.Open(os.path.join(FLAGS.save_dir, 'val.txt'), 'w') as val_f:
          logging.info('Generating data...')
          for frame in all_frames:
            # actually generating images and masks
            _gen_example(frame, all_examples)

          for _, example in all_examples.items():
            if example:
              s = example['folder_name']
              frame = example['file_name']
              if np.random.random() < 0.1:
                val_f.write('%s %s\n' % (s, frame))
              else:
                train_f.write('%s %s\n' % (s, frame))

  else:
      frame_chunks = np.array_split(all_frames, NUM_CHUNKS)
      manager = multiprocessing.Manager()
      all_examples = manager.dict()
      num_cores = multiprocessing.cpu_count()
      num_threads = num_cores if FLAGS.num_threads is None else FLAGS.num_threads
      pool = multiprocessing.Pool(num_threads)

      if not gfile.Exists(FLAGS.save_dir):
        gfile.MakeDirs(FLAGS.save_dir)

      with gfile.Open(os.path.join(FLAGS.save_dir, 'train.txt'), 'w') as train_f:
        with gfile.Open(os.path.join(FLAGS.save_dir, 'val.txt'), 'w') as val_f:
          logging.info('Generating data...')
          for index, frame_chunk in enumerate(frame_chunks):
            all_examples.clear()
            pool.map(_gen_example_star,
                     zip(frame_chunk, itertools.repeat(all_examples)))
            logging.info('Chunk %d/%d: saving %s entries...', index + 1, NUM_CHUNKS,
                         len(all_examples))
            for _, example in all_examples.items():
              if example:
                s = example['folder_name']
                frame = example['file_name']
                if np.random.random() < 0.1:
                  val_f.write('%s %s\n' % (s, frame))
                else:
                  train_f.write('%s %s\n' % (s, frame))
      pool.close()
      pool.join()
  if FLAGS.dataset_name=='video' and FLAGS.delete_temp:
      dataloader.delete_temp_images()
  
  
  
def _gen_example(i, all_examples):
  """Saves one example to file.  Also adds it to all_examples dict."""
  example = dataloader.get_example_with_index(i)
  if not example:
    return
  image_seq_stack = _stack_image_seq(example['image_seq'])
  example.pop('image_seq', None)  # Free up memory.
  intrinsics = example['intrinsics']
  fx = intrinsics[0, 0]
  fy = intrinsics[1, 1]
  cx = intrinsics[0, 2]
  cy = intrinsics[1, 2]
  save_dir = os.path.join(FLAGS.save_dir, example['folder_name'])
  if not gfile.Exists(save_dir):
    gfile.MakeDirs(save_dir)
  img_filepath = os.path.join(save_dir, f'{example["file_name"]}.{FLAGS.save_img_ext}')
  imageio.imsave(img_filepath, image_seq_stack.astype(np.uint8))

  if FLAGS.gen_mask:
      mask_seq_stack = _stack_image_seq(example['mask_seq'])
      example.pop('mask_seq', None)  # Free up memory.
      mask_filepath = os.path.join(save_dir,
                                   f'{example["file_name"]}-fseg.{FLAGS.save_img_ext}')
      imageio.imsave(mask_filepath, mask_seq_stack.astype(np.uint8))

  cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
  example['cam'] = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy)
  with open(cam_filepath, 'w') as cam_f:
    cam_f.write(example['cam'])

  key = example['folder_name'] + '_' + example['file_name']
  all_examples[key] = example


def _gen_example_star(params):
  return _gen_example(*params)


def _stack_image_seq(seq):
  for i, im in enumerate(seq):
    if i == 0:
      res = im
    else:
      res = np.hstack((res, im))
  return res


def main(_):
  _generate_data()


if __name__ == '__main__':
  app.run(main)
