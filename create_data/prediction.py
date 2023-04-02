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
import os, glob, sys

from dsr_tree_project.create_data.utils import download_image_tiles_from_ee


model_weights = '../models/Detectron2/model_final.pth'
yaml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def get_predictor(model_weights=model_weights, score_threshold=0.1, nms_threshold=0.2, device='cpu'):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    return DefaultPredictor(cfg)

def predict_trees_for_area(longitude, latitude, predictor, tile_width=1024, tile_height=640, rows=3, cols=3, tmp_dir='/tmp', overlap=0):
    #connect to gee before
    download_image_tiles_from_ee(center = [longitude, latitude], 
                                 rows=rows, 
                                 cols=cols, 
                                 img_dim=[tile_width, tile_height], 
                                 rel_overlap=overlap,
                                 out_dir = tmp_dir
                                 )
    list_tiles = [os.path.basename(x) for x in glob.glob(f"{tmp_dir}/*.tif")]
    
    df_list = []
    i = 0
    for tile in list_tiles:
        tiff_path = os.path.join(tmp_dir, tile)
        df_tile = predict_trees_for_tile(tiff_path, predictor)
        df_list.append(df_tile)

        i += 1
        print('Predicted ' + str(i) + ' of ' + str(len(list_tiles)) + ' tiles.')
    return pd.concat(df_list, ignore_index=True)

def predict_trees_for_tile(tiff_path, predictor):
    im = cv2.imread(tiff_path)
    tiff = rasterio.open(tiff_path)
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

