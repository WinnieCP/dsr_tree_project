
import streamlit as st
from streamlit_folium import folium_static
import geemap.eefolium as geemap
import ee
import json


# Data from the downloaded JSON file
json_data = st.secrets["json_data"]
service_account = st.secrets["service_account"]

# Preparing values
json_object = json.loads(json_data, strict=False)
json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

st.title('Trees of Berlin')

with st.echo():
    import streamlit as st
    from streamlit_folium import folium_static
    import geemap.eefolium as geemap
    import ee

    map = geemap.Map()
    image = ee.Image("Germany/Brandenburg/orthos/20cm").select(['R','G','B'])
    map.addLayer(image, {}, "Earth Engine")



    map.addLayerControl()

    # call to render Folium map in Streamlit
    folium_static(map)