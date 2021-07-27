#!/bin/bash
# Download the ResNet-18 torch checkpoint
if [ -e ./ckpt_prep/resnet-18.t7 ]
then
    echo 'file already exists'
else
    echo 'downloading resnet-18 torch checkpoint'
    wget -P ./ckpt_prep https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
fi

# Convert into tensorflow checkpoint
cd ./ckpt_prep
python ./extract_torch_t7.py
