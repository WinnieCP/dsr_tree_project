import pandas as pd
import numpy as np


# you may need !pip install rasterio
import rasterio
#import rasterio.plot
import rasterio.features
import rasterio.warp


def add_new_crs_to_df(df, crs):
    '''
    input: dataframe with columns X and Y representing longitude and latitude,
        new coordinate reference system
    output: dataframe with additional columns X_crs and Y_crs  
    '''
    df['X_crs'], df['Y_crs'] = rasterio.warp.transform('EPSG:4326', crs, np.array(df.loc[:,'X']), np.array(df.loc[:,'Y']))
    return df


def compute_center_point(tiff, df):
    '''
    input: geotif image
        Dataframe of baumktaster with matching crs columns X_crs and Y_crs and kronedurch
    output: array of center point of trees in pixel
    '''
    idy, idx = tiff.index(df['X_crs'], df['Y_crs'], precision=1E-6)
    return np.array(idx), np.array(idy)


def compute_wh(diameter, resolution):
    '''
    input: resoultion in m (e.g 0.2), diameter in m (can be a single value or array)
    output: width/height in pixel
    '''
    return (np.round(diameter / resolution)).astype(int)


def compute_bbox_xywh(tiff, df):
    '''
    input: geotif image
        Dataframe of baumktaster with matching crs columns X_crs and Y_crs and kronedurch
    output: array of x center, y center, width, height of bounding box in pixel
    '''
    trees_in_picture = df.loc[(df.X_crs > tiff.bounds[0] ) & (df.X_crs < tiff.bounds[2] ) & 
                              (df.Y_crs > tiff.bounds[1] ) & (df.Y_crs < tiff.bounds[3] )]
    idx, idy = np.array( compute_center_point(tiff, trees_in_picture))
    d = compute_wh(np.array(trees_in_picture.kronedurch), tiff.transform.a)
    mask = ((idx - d/2 >= 0) & (idx + d/2 <= tiff.width) & 
            (idy - d/2 >= 0) & (idy + d/2 <= tiff.height))
    return idx[mask], idy[mask], d[mask], d[mask]


def compute_bbox_xyxy(tiff, df):
    '''
    input: geotif image
        Dataframe of baumktaster with matching crs columns X_crs and Y_crs and kronedurch
    output: array of x_min, y_min, x_max, y_max values of bounding box in pixel
    '''
    trees_in_picture = df.loc[(df.X_crs > tiff.bounds[0] ) & (df.X_crs < tiff.bounds[2] ) & 
                              (df.Y_crs > tiff.bounds[1] ) & (df.Y_crs < tiff.bounds[3] )]
    idx, idy = compute_center_point(tiff, trees_in_picture)
    r = (np.floor(compute_wh(np.array(trees_in_picture.kronedurch), tiff.transform.a) / 2)).astype(int)
    mask = ((idx - r >= 0) & (idx + r <= tiff.width) & 
            (idy - r >= 0) & (idy + r <= tiff.height))
    return (idx - r) [mask], (idy - r)[mask], (idx + r)[mask], (idy + r)[mask]
