import pandas as pd
import numpy as np
import ee
import requests
import shutil
import os
# you may need !pip install rasterio
import rasterio
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


def download_image_tiles_from_ee(center, 
                                 rows=2, 
                                 cols=2, 
                                 img_dim=640, 
                                 rel_overlap=0,
                                 scale=1,
                                 out_dir = '/.',
                                 channels = ['R', 'G', 'B'],
                                 processes = 25,
                                 format = 'GEO_TIFF',
                                 prefix = 'tile_'
                                 ):
    '''
    Downloads a grid of tiles from the Google Earth engine Image 'Germany/Brandenburg/orthos/20cm'
    and saves them as geotif to a specified folder
    input:
        center:     (float, float), (longitude, latitude) of point around which to sample tiles
        rows:       int, number of rows (latitude) to sample
        cols:       int, number of columns (longitude) to sample
        img_dims:   int or list of 2 ints specifying the width and height of a tile in pixels
        rel_overlap: float in [0,0.5], neighboring tiles will overlap this much of their width or height
        scale:      float >1, how much to rescale the image resolution. 1: original scale, 
                    n: n original pixels are summarized into one
        out_dir:    str, path to folder where to save the generated tiles
        channels:   list of str, entry can be 'R', 'G', 'B', 'N' 
        format:     string, 'GEO_TIFF' or 'png','jpg'
    '''

    # check input
    assert center[0]<center[1], 'For Berlin/Brandenburg, longitude < latitude. Likely you have given the center coordinates in the wrong order.'
    assert rel_overlap>=0 and rel_overlap<=0.5, f'Relative overlap needs to be in [0,0.5]. You set rel_overlap={rel_overlap}'
    assert scale>=1, f'Scale needs to be larger than 1. You set scale={scale}.'
    # assert that only RNGB are in channels

    # infer image dimensions if only a single value is given (for square images)
    img_dim_error = 'img_dim needs to be either a list with 2 float or int entries or a single float or int (for square images)'
    if type(img_dim)!=list:
        assert type(img_dim)==float or type(img_dim==int), img_dim_error
        img_dim=[img_dim, img_dim]
    else:
        assert len(img_dim)==2, img_dim_error
    
    # specify the image from Earth Engine from which to generate the tiles
    image = (
        ee.Image("Germany/Brandenburg/orthos/20cm")
        .select(channels)
    )

    params = {'rows' : rows,
                'cols' : cols,
                'img_dim' : img_dim,
                'rel_overlap' : rel_overlap,
                'scale' : scale,
                'out_dir' : out_dir,
                'center' : center,
                'channels' : channels,
                'processes' : processes,
                'format' : format,
                'prefix' : prefix,
                'dimensions': f'{img_dim[0]}x{img_dim[1]}',
                'count' : cols*rows
                }
    
    tiles = get_grid_tiles(image, params)
    if len(tiles)==1:
      getResult(0, params, tiles[0])
    else:
      processes = params['processes']
      if len(tiles) < processes:
        processes = len(tiles)
      print(f'starting {processes} processes')
      pool = multiprocessing.Pool(processes)
      pool.starmap(getResult, zip(range(len(tiles)),[params]*len(tiles), tiles)) # this might not work
      pool.close()
    return tiles, params
    
def get_grid_tiles(image, params):

    p = SimpleNamespace(**params)

    # create a grid in meters around the center point
    projection = image.projection()
    img_scale = projection.nominalScale().getInfo() # width of 1 pixel
    grid_step_in_m = [dim * (1 - p.rel_overlap) * img_scale for dim in p.img_dim]

    center_point = ee.Geometry.Point(p.center).transform(projection.crs()).getInfo()['coordinates']
    lon_list = center_point[0]+(np.arange(p.cols)-p.cols/2+1/2)*grid_step_in_m[0]
    lat_list = center_point[1]+(np.arange(p.rows)-p.rows/2+1/2)*grid_step_in_m[1]
    point_list = np.array(np.meshgrid(lon_list, lat_list)).reshape(2,-1).T
    tiles = [ee.Geometry.Polygon(
      [
        [
            [lon-grid_step_in_m[0]/2, lat-grid_step_in_m[1]/2],
            [lon-grid_step_in_m[0]/2, lat+grid_step_in_m[1]/2],
            [lon+grid_step_in_m[0]/2, lat+grid_step_in_m[1]/2],
            [lon+grid_step_in_m[0]/2, lat-grid_step_in_m[1]/2],
            [lon-grid_step_in_m[0]/2, lat-grid_step_in_m[1]/2],
        ],
      ],
      evenOdd = False,
      proj=projection.crs()
      ) for lon, lat in point_list]

    return tiles

@retry(tries=10, delay=1, backoff=2)

def getResult(index, params, tile):
    if params['format'] in ['png', 'jpg']:
        url = image.getThumbURL(
            {
                'region': tile,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )
    else:
        url = image.getDownloadURL(
            {
                'region': tile,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )

    if params['format'] == "GEO_TIFF":
        ext = 'tif'
    else:
        ext = params['format']

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    out_dir = os.path.abspath(params['out_dir'])
    basename = str(index).zfill(len(str(params['count'])))
    filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", basename)

