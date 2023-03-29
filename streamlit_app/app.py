
import streamlit as st
from streamlit_folium import folium_static
import geemap.eefolium as geemap
import ee
from google.oauth2 import service_account
from ee import oauth


def get_auth():
    service_account_keys = st.secrets['ee_keys']
    credentials = service_account.Credentials.from_service_account_info(
        service_account_keys, scopes=oauth.SCOPES)
    ee.Initialize(credentials)
    return 'successfully sync to GEE'
    
# os.environ["EARTHENGINE_TOKEN"] == st.secrets["EARTHENGINE_TOKEN"]

"# streamlit geemap demo"
#st.markdown('Source code: <https://github.com/giswqs/geemap-streamlit/blob/main/geemap_app.py>')
st.title('Trees of Berlin')

with st.echo():
    import streamlit as st
    from streamlit_folium import folium_static
    import geemap.eefolium as geemap
    import ee

    map = geemap.Map()
    image = ee.Image("Germany/Brandenburg/orthos/20cm")
        .select(['R','G','B'])
    map.addLayer(image, {}, "Earth Engine")



    map.addLayerControl()

    # call to render Folium map in Streamlit
    folium_static(map)