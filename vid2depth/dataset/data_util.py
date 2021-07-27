import os
import sys
import numpy as np
import skimage
import cv2

# Root directory of the RCNN repository
ROOT_DIR = os.path.abspath("../Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def gen_mask(images):
    if isinstance(images, list):
        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        config = InferenceConfig()

        # Create model object in inference mode.
        model =  modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        masks = []
        for image in images: 
            results = model.detect([image], verbose=1)
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
            masks.append(mask_img)
        return masks

