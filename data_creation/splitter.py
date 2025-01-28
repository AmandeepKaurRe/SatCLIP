import geopandas as gpd
import os

# Read the GeoJSON file
def split_gdfs(file_name, chunk_size):
    direc = os.path.dirname(file_name)
    gdf = gpd.read_file(file_name)
    gdf_size = gdf.shape[0]
    os.mkdir(f"{direc}/split_geojsons")
    for i in range((gdf_size//chunk_size)+1):
        new_gdf = gdf.iloc[i*chunk_size:(i+1)*chunk_size]
        new_gdf.to_file(f"{direc}/split_geojsons/{i*chunk_size}-{(i+1)*chunk_size}.geojson", driver="GeoJSON")

file_name = "/home/akaur64/GeoSSL/SATCLIP/data_creation/125000_18000.geojson"
split_gdfs(file_name, 2000)