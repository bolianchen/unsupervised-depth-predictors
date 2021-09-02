# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""A binary for generating odometry trajectories given a checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import threading
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from depth_from_video_in_the_wild import model
import cv2

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import PIL.Image as pil

flags.DEFINE_string('test_file_dir', None,
                    'Directory where the test images are')
flags.DEFINE_string(
    'output_dir', None, 'Directory to store predictions. '
    'Subdirectories will be created for each checkpoint.')
flags.DEFINE_string(
    'file_extension', 'jpg', 'format of the saved images')
flags.DEFINE_string('checkpoint_path', None,
                    'path of the checkpoint to evaluate')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_bool('output_img_disp', False, 'save depth in picture format')
flags.DEFINE_bool('output_npy', False, 'save depth in numeric format')

# options below only effective if there are multiple test images in test_file_dir
flags.DEFINE_bool('output_anime', False, 'save gif')
flags.DEFINE_integer('max_count', 0, 'maximum number of images to do inference')


FLAGS = flags.FLAGS


def depth_inference():
    """Generates trajectories from the KITTI odometry sets and a checkpoint."""

    batch_size = 1
    # Note that the struct2depth code only works at batch_size=1, because it uses
    # the training mode of batchnorm at inference.
    inference_model = model.Model(is_training=False,
                                  batch_size=batch_size,
                                  img_height=FLAGS.img_height,
                                  img_width=FLAGS.img_width)
    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, FLAGS.checkpoint_path)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    im_files = []
    # collect all the images in the test_file_dir
    for any_file in tf.gfile.ListDirectory(FLAGS.test_file_dir):
        if 'seg' in any_file:
            continue
        if '.png' in any_file:
            im_files.append(os.path.join(FLAGS.test_file_dir, any_file))
        if '.jpg' in any_file:
            im_files.append(os.path.join(FLAGS.test_file_dir, any_file))
    im_files = sorted(im_files)

    if FLAGS.max_count:
        im_files = im_files[:FLAGS.max_count]

    im_batch = []
    if FLAGS.output_anime:
        multi_img_disp = []
        anime_name = os.path.basename(FLAGS.test_file_dir) + '.gif'
    for i in range(len(im_files)):
        if i % 100 == 0:
            logging.info('%s of %s files processed.', i, len(im_files))
        # Read image and run inference.
        im = load_image(im_files[i],
                        resize=(FLAGS.img_width, FLAGS.img_height))

        im_batch.append(im)

        if len(im_batch) == batch_size or i == len(im_files) - 1:
            # Call inference on batch.
            for _ in range(batch_size - len(im_batch)):  # Fill up batch.
                im_batch.append(
                    np.zeros(shape=(img_height, img_width, 3),
                             dtype=np.float32))
            im_batch = np.stack(im_batch, axis=0)
            est_depth = inference_model.inference_depth(im_batch, sess)

            for j in range(len(im_batch)):
                color_map = normalize_depth_for_display(
                    np.squeeze(est_depth[j]))
                visualization = np.concatenate((im_batch[j], color_map),
                                               axis=0)
                # Save raw prediction and color visualization. Extract filename
                # without extension from full path: e.g. path/to/input_dir/folder1/
                # file1.png -> file1
                k = i - len(im_batch) + 1 + j
                filename_root = os.path.splitext(os.path.basename(
                    im_files[k]))[0]
                output_vis = os.path.join(
                        FLAGS.output_dir,
                        filename_root + '.' +FLAGS.file_extension)
                        
                if FLAGS.output_npy:
                    output_raw = os.path.join(FLAGS.output_dir,
                                              filename_root + '.npy')
                    with tf.gfile.Open(output_raw, 'wb') as f:
                        np.save(f, est_depth[j])

                if FLAGS.output_img_disp:
                    save_image(output_vis, visualization, FLAGS.file_extension)
                if FLAGS.output_anime:
                    multi_img_disp.append(
                            pil.fromarray((visualization*255).astype(np.uint8))
                            )
                
            im_batch = []

    if FLAGS.output_anime and len(im_files)>1:
        multi_img_disp[0].save(os.path.join(FLAGS.output_dir, anime_name),
                      save_all=True,
                      append_images=multi_img_disp[1:],
                      duration=40,
                      loop=0)

def load_image(img_file, resize=None, interpolation='linear'):
    """Load image from disk. Output value range: [0,1]."""
    with tf.gfile.Open(img_file, 'rb') as f:
        im_data = np.fromstring(f.read(), np.uint8)
    im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if resize and resize != im.shape[:2]:
        ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
        im = cv2.resize(im, resize, interpolation=ip)
    return im.astype(np.float32) / 255.0


def normalize_depth_for_display(depth,
                                pc=95,
                                crop_percent=0,
                                normalizer=None,
                                cmap='plasma'):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.

    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    return disp


def save_image(img_file, im, file_extension):
    """Save image from disk. Expected input value range: [0,1]."""
    im = (im * 255.0).astype(np.uint8)
    with tf.gfile.Open(img_file, 'w') as f:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        _, im_data = cv2.imencode('.%s' % FLAGS.file_extension, im)
        f.write(im_data.tostring())


def gray2rgb(im, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def main(_):
    depth_inference()


if __name__ == '__main__':
    app.run(main)
