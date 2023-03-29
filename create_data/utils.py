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
from retry import retry
from types import SimpleNamespace
import multiprocessing

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
                                 prefix = 'tile_',
                                 preview_only = False,
                                 min_num_trees = None,
                                 path_to_tree_data = None,
                                 min_kronedurch = 0,
                                 only_tiles_with_medium_amount_of_labels = False
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
        prefix:     prefix of the downloaded image tiles
        preview_only: No download, just returns the tiles for visualizing,
        min_num_trees: int, if set only those tiles which contain at least 
                        min_num_trees trees are returned
        min_kronedurch: float, if not None, only trees that have an entry for kronedurch which is
                        larger than min_kronedurch are considered in the filtering of tiles
        path_to_tree_data: str, path to where the Baumkataster csv lies
        only_tiles_with_medium_amount_of_labels: filters tiles and returns only those that have a
                    number of trees from the 10-90%tile of the distribution in the selected area

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

    if min_num_trees is not None:
        assert path_to_tree_data is not None, 'If min_num_trees is set, path_to_tree_data must also be given.'
        assert not only_tiles_with_medium_amount_of_labels, 'If min_num_trees is set, only_tiles_with_medium_amount_of_labels needs to be False.'

    if only_tiles_with_medium_amount_of_labels:
        assert path_to_tree_data is not None, 'If min_num_trees is set, path_to_tree_data must also be given.'
        assert min_num_trees is None, 'If only_tiles_with_medium_amount_of_labels, then min_num_trees needs to be None.'

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
                'count' : cols*rows,
                'min_num_trees' : min_num_trees,
                'path_to_tree_data' : path_to_tree_data,
                'min_kronedurch' : min_kronedurch,
                'only_tiles_with_medium_amount_of_labels' : only_tiles_with_medium_amount_of_labels
                }
    
    # specify the image from Earth Engine from which to generate the tiles
    image = (
        ee.Image("Germany/Brandenburg/orthos/20cm")
        .select(channels)
    )

    tiles = get_grid_tiles(image, params)
    
    # download the tiles if required
    if not preview_only:
        if len(tiles)==1:
            getResult(0, params, tiles[0])
        else:
            processes = params['processes']
            if len(tiles) < processes:
                processes = len(tiles)
            print(f'starting {processes} processes')
            pool = multiprocessing.Pool(processes)
            pool.starmap(getResult, zip([image]*len(tiles), range(len(tiles)),[params]*len(tiles), tiles)) # this might not work
            pool.close()

    return tiles, params
    
def get_grid_tiles(image, params):

    # make all variables in params accessible through p.<variable name>
    p = SimpleNamespace(**params) 

    # get properties of the image projection
    projection = image.projection()
    img_scale = projection.nominalScale().getInfo() # width of 1 pixel

    # convert lon, lat coordinates of center to crs of image
    center_point = ee.Geometry.Point(p.center).transform(projection.crs()).getInfo()['coordinates']

    # create a grid of points around the given center point (in meters)
    # each of these points will be the center of a (potential) tile
    grid_step_in_m = [dim * (1 - p.rel_overlap) * img_scale for dim in p.img_dim]
    lon_list = center_point[0]+(np.arange(p.cols)-p.cols/2+1/2)*grid_step_in_m[0]
    lat_list = center_point[1]+(np.arange(p.rows)-p.rows/2+1/2)*grid_step_in_m[1]
    point_list = np.array(np.meshgrid(lon_list, lat_list)).reshape(2,-1).T

    # if required, filter out points around which the tile does not 
    # include sufficient number of trees
    if p.min_num_trees is not None or p.only_tiles_with_medium_amount_of_labels:
        # Determine number of trees per tile
        trees_per_tile = get_trees_per_tile(point_list, projection,
                                             grid_step_in_m, lon_list, 
                                             lat_list, params)
        # filter the list of potential tile centers based on the requirements on 
        # number of trees in the tile
        point_list = filter_tiles(point_list, trees_per_tile, params)

    # generate the tiles
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

def filter_tiles(point_list, trees_per_tile, params):

    # make all variables in params accessible through p.<variable name>
    p = SimpleNamespace(**params)

    if p.min_num_trees is not None:
        # filter based on minimal required number of trees per tile
        point_list = point_list[trees_per_tile>=p.min_num_trees]

    if p.only_tiles_with_medium_amount_of_labels:
        # filter such that the tiles with a tree count
        # in the upper and lower 10% of the distribution
        # of number of trees per tile is excluded
        # (ignoring tree count 0 in the generation of 
        # the distribution)
        cutoff_limits = np.quantile(trees_per_tile[trees_per_tile>0],[0.1,0.9])
        point_list = point_list[(trees_per_tile<=cutoff_limits[1])&
                                (trees_per_tile>=cutoff_limits[0])]
    return point_list

def get_trees_per_tile(pt_list, projection, grid_step_in_m, lon_list, lat_list, params):

    # make all variables in params accessible through p.<variable name>
    p = SimpleNamespace(**params)

    # read the Baumkataster data
    df = pd.read_csv(p.path_to_tree_data)
    if p.min_kronedurch is not None:
        df = df[df['kronedurch'] >= p.min_kronedurch]

    # add tree positions in the crs of the image
    df = add_new_crs_to_df(df, projection.crs().getInfo())[['X_crs','Y_crs','kronedurch']]

    # bin the longitude and latitude of each tree in the new crs with bins
    # given by the tile grid
    bins_lon = np.hstack([lon_list - grid_step_in_m[0] / 2, lon_list[-1] + grid_step_in_m[0]/2])
    bins_lat = np.hstack([lat_list - grid_step_in_m[1] / 2, lat_list[-1] + grid_step_in_m[1]/2]) 
    df['tile_x'] = pd.cut(df['X_crs'], bins = bins_lon, labels = range(len(bins_lon)-1))
    df['tile_y'] = pd.cut(df['Y_crs'], bins = bins_lat, labels = range(len(bins_lat)-1))

    # determine the number of trees per tile
    df = df.groupby(['tile_x','tile_y']).count()

    trees_per_tile = np.array([df.loc[(np.argwhere(lon_list==lon)[0][0],
                                       np.argwhere(lat_list==lat)[0][0]),'X_crs']
                                       for lon, lat in pt_list])
    return trees_per_tile

@retry(tries=10, delay=1, backoff=2)
def getResult(image, index, params, tile):
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


