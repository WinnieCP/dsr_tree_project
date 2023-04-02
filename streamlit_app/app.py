
import streamlit as st
from streamlit_folium import folium_static, st_folium
import geemap.foliumap as geemap
import ee
import json


json_data = st.secrets["json_data"]
service_account = st.secrets["service_account"]

# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

st.title('Trees of Berlin')



map = geemap.Map(basemap='HYBRID')
image = ee.Image("Germany/Brandenburg/orthos/20cm").select(['R','G','B'])
map.addLayer(image, {}, "Earth Engine")



map.addLayerControl()

# call to render Folium map in Streamlit
# folium_static(map)

map_info = st_folium(map, width=700, height=700)

if st.button("detect trees", type="primary"):  
    # download images around current center of map
    # predict images with the model
    # display detected bounding boxes

    st.write(map_info)