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

#!/bin/bash
set -e # stop the execution of a script if a command or pipeline has an error
set -x # print all executed commands to the terminal

python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=../test_motion \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "KITTI_processed/train.txt"
      }
    },
    "trainer": {
      "init_ckpt": "Imagenet_ckpt/model.ckpt",
      "init_ckpt_type": "imagenet",
    }
  }'
