
import geemap.foliumap as geemap
import ee
import json
import pandas as pd

def add_eelayer( map, channels=['R', 'G', 'B'],
                 ee_name = "Germany/Brandenburg/orthos/20cm"):
    img = ee.Image(ee_name).select(channels)
    map.addLayer(img, {}, "Earth Engine "+str(channels))
    return img

path_to_secret = '../secrets/trees-in-berlin_for_streamlit.json'

# Preparing values
with open(path_to_secret, 'r') as json_data:
    json_object = json.loads(json_data.read(), strict=False)
    service_account = json_object['client_email']
    json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

center_pt = [13.420409,52.534824]#[13.42378, 52.535262]
map = geemap.Map()
map.setCenter(center_pt[0],center_pt[1],18)

# option to display also nrg image as layer:
image_nrg = add_eelayer(map, channels=['N','R','G'])


# add earth engine image to map
image_rgb = add_eelayer(map)

# add black white layer (might be easier on the eye with all the annotations)

image_bw = add_eelayer(map, channels=['G'])

# get projection of ee image
projection = image_rgb.projection()

# add validation labels
df_reg = pd.read_csv('../data/Baumkataster_Prenzlauer_Berg.csv')
registry_label_style = {'color': 'ffffffff ', 'width': 4, 'lineType': 'solid', 'fillColor': 'ffffff00'}
registry_labels = ee.FeatureCollection([ee.Geometry.Point((x,y)).buffer(d/2).bounds() 
                              for x, y, d in zip(df_reg.X, 
                                               df_reg.Y,
                                               df_reg.kronedurch)])
map.addLayer(registry_labels.style(**registry_label_style),{}, 'tree registry', shown=False)


# add Detectron2 labels
df_d2 = pd.read_csv('../data/D2_pred_Prenzlauer_Berg.csv')
d2_label_style = {'color': 'ff00a9ff', 'width': 4, 'lineType': 'solid', 'fillColor': '00000000'}
d2_labels = ee.FeatureCollection([ee.Geometry.Point((x,y)).buffer(d/2).bounds() 
                               for x, y, d in zip(df_d2.X, 
                                                df_d2.Y,
                                               df_d2.kronedurch)])
map.addLayer(d2_labels.style(**d2_label_style),{}, 'Detectron2', shown=False)

#d2_label_style = {'color': 'ff00a9ff', 'width': 3, 'lineType': 'solid', 'fillColor': '00000000'}
#map.addLayer(d2_labels.style(**d2_label_style),{}, 'Detectron2 width3', shown=False)

#d2_label_style = {'color': 'ff00a9ff', 'width': 2, 'lineType': 'solid', 'fillColor': '00000000'}
#map.addLayer(d2_labels.style(**d2_label_style),{}, 'Detectron2 width2', shown=False)


# add YOLO labels
df_yolo = pd.read_csv('../data/Yolo_pred_Prenzlauer_Berg.csv')
yolo_label_style = {'color': 'ffbf00ff ', 'width': 4, 'lineType': 'solid', 'fillColor': '00000000'}
yolo_labels = ee.FeatureCollection([ee.Geometry.Point((x,y)).buffer(d/2).bounds() 
                               for x, y, d in zip(df_yolo.X, 
                                                df_yolo.Y,
                                               df_yolo.kronedurch)])
map.addLayer(yolo_labels.style(**yolo_label_style),{}, 'YOLO', shown=False)

# add layer control
map.addLayerControl()

map.save('tree_map.html')