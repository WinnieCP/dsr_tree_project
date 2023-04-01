#!pip install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
#!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#or
#!git clone https://github.com/facebookresearch/detectron2.git
#!python -m pip install -e detectron2

import numpy as np
import pandas as pd

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import rasterio
import rasterio.warp
import cv2

score_threshold = .1
nms_threshold = 0.2
model_weights = '../data/model_final.pth'
yaml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def get_predictor(model_weights=model_weights, score_threshold=score_threshold, nms_threshold=nms_threshold):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    return DefaultPredictor(cfg)

def predict_trees(longitude, latitude, predictor):
    im = cv2.imread('../data/tiles/tile_10.tif')
    tiff = rasterio.open('../data/tiles/tile_10.tif')
    bbox = predict_bbox(im, predictor)
    df = bbox_to_baumkataster(bbox, tiff)
    return df

def predict_bbox(image, predictor):
    pred = predictor(image)
    return pred['instances'].pred_boxes.tensor.numpy()

def bbox_to_baumkataster(bbox, tiff):
    center = np.array([bbox[:,2] + bbox[:,0], bbox[:,3] + bbox[:,1]]) / 2
    diameter = np.mean(np.abs(np.array([bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]])), axis=0)

    center_x, center_y = tiff.xy(center[1,:], center[0,:])
    diameter = diameter * tiff.transform.a

    center_x, center_y = rasterio.warp.transform(tiff.crs, 'EPSG:4326', center_x, center_y)

    df_tile = pd.DataFrame(data={'X': center_x, 'Y': center_y, 'kronedurch': diameter})
    return df_tile