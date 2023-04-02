
import streamlit as st
from streamlit_folium import folium_static #st_folium
import geemap.foliumap as geemap
import ee
import json
import pandas as pd

from prediction import get_predictor, predict_trees_for_tile, predict_trees_for_area

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
#ee.Initialize(credentials)


st.title('Trees of Berlin')

center_pt = [13.42378, 52.535262]
map = geemap.Map()
map.setCenter(center_pt[0],center_pt[1],17)

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
#map.addLayer(validation_labels.style(**val_label_style),{}, 'validation dataset')


predict_button = st.button("Predict trees")


first_prediction = True
if predict_button:  
    #write center of map 
    #possible using st_folium
    #center = st_data['center']
    #st.write(center)

    #load predictor 
    if first_prediction:
        predictor = get_predictor()
        st.write('Load predictor')
    first_prediction = False

    #predict for a fixed tile
    #df_pred = predict_trees_for_tile('../data/tiles/tile_10.tif',predictor)
    df_pred = predict_trees_for_area(center_pt[0], center_pt[1], predictor)

    pred_label_style = {'color': '03fce3ff ', 'width': 2, 'lineType': 'solid', 'fillColor': '00000000'}
    prediction_labels = ee.FeatureCollection([ee.Geometry.Point((x,y)).buffer(d/2).bounds() 
                              for x, y, d in zip(df_pred.X, df_pred.Y, df_pred.kronedurch)])
    map.addLayer(prediction_labels.style(**pred_label_style),{}, 'predictions')


# add layer control
map.addLayerControl()

# call to render Folium map in Streamlit
folium_static(map)
#st_data = st_folium(map, width=3000) 
#st_folium offers information about current location in map
#but the location st_folium displays is shifted :(
