
import streamlit as st
from streamlit_folium import folium_static
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


st.title('Trees of Berlin')

center_pt = [13.42378, 52.535262]
map = geemap.Map()
map.setCenter(center_pt[0],center_pt[1],17)

# option to display also nrg image as layer:
# image_nrg = add_eelayer(map, channels=['N','R','G'])

# add earth engine image to map
image_rgb = add_eelayer(map)

# get projection of ee image
projection = image_rgb.projection()

# add validation labels
df_val = pd.read_csv('../data/Val_Baumkataster.csv')
val_label_style = {'color': '03fce3ff ', 'width': 2, 'lineType': 'solid', 'fillColor': '00000000'}
validation_labels = ee.FeatureCollection([ee.Geometry.Point((x,y), proj=projection.crs()).buffer(d/2).bounds() 
                              for x, y, d in zip(df_val.X_crs, 
                                               df_val.Y_crs,
                                               df_val.kronedurch)])
map.addLayer(validation_labels.style(**val_label_style),{}, 'validation dataset')


# add training labels
df_train = pd.read_csv('../data/Train_Baumkataster.csv')
train_label_style = {'color': 'fffce3ff ', 'width': 2, 'lineType': 'solid', 'fillColor': '00000000'}
training_labels = ee.FeatureCollection([ee.Geometry.Point((x,y), proj=projection.crs()).buffer(d/2).bounds() 
                              for x, y, d in zip(df_train.X_crs, 
                                               df_train.Y_crs,
                                               df_train.kronedurch)])
map.addLayer(training_labels.style(**train_label_style),{}, 'training dataset')

# add layer control
map.addLayerControl()

# call to render Folium map in Streamlit
folium_static(map)

if st.button("detect trees", type="primary"):  
    # download images around current center of map
    # predict images with the model
    # display detected bounding boxes

    st.write('running model...')
