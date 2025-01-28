from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc

import geopandas as gpd
from shapely.geometry import Point
import random
import pandas as pd
import multiprocessing as mp

# Import libraries
import geopandas as gpd
from shapely.geometry import Point, Polygon
import random
import pandas as pd

import pystac_client
import planetary_computer
from pystac.extensions.eo import EOExtension as eo

import stackstac
import warnings
import numpy as np
import rioxarray
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
import io
import os
import time

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import numpy as np


random.seed(0)
np.random.seed(0)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


import psutil
import os

def log_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    with open(LOG_FILE, "a") as f:  # Append mode ensures data isn't overwritten
        f.write(f"{mem:.2f}, {pid}, {datetime.now()}\n")

# This function will take lon/lat, and write to file the xarry patch sampled
def patch_sampler(lon, lat, continent, save_params=None):
    # try:
    area_of_interest = {"type": "Point", "coordinates": [lon, lat]} # example for rwanda point: lon=29.9309, lat = -1.9154

    search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=DATE_RANGE,
    query={"eo:cloud_cover": {"lt": CLOUD_COVER}},
    )

    # Check how many items were returned
    items = search.item_collection()

    # in the case there are no collections found
    if len(items)==0:
        # print("no collections found")
        return False, []

    # look for leas cloudy from the collection
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    # Create a stack with required Sentinel Bands (B1 to B12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack = stackstac.stack(
            least_cloudy_item,
            assets=[
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                # "B10", #added
                "B11",
                "B12",
            ],
        )
    _, num_channels, height, width = stack.shape

    x = np.random.randint(0, width - PATCH_SIZE[0])
    y = np.random.randint(0, height - PATCH_SIZE[1])

    patch = stack[0, :, y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]].compute()
    percent_empty = np.mean((np.isnan(patch.data)).sum(axis=0) == num_channels)

    if save_params:
        dir = os.path.join(save_params["output_dir"],save_params["continent"])
        if not os.path.exists(dir):
            os.makedirs(dir)
        f_name = "patch_"+str(save_params["sample_id"])+".tif"
        patch_path = dir = os.path.join(dir, f_name)
    else:
        patch_path = "sample_test.tif"

    if percent_empty > 0.1:
        if REPORT_BAD:
            with open(TXT_FILE, "a") as f:  # Append mode ensures data isn't overwritten
                f.write(f"Empty {save_params['sample_id']}\n")
        gc.collect()
        return False, []
    else:
        raster_path = patch_path
        patch.values = np.nan_to_num(patch.values)
        patch.values = patch.values.astype(np.uint16)  # Ensure correct data type
        patch.rio.to_raster(raster_path=patch_path, driver="COG", dtype=np.uint16, compress="LZW", nodata=0, predictor=2)
        bounds = stackstac.array_bounds(patch, to_epsg=4326)
        x = (bounds[0]+bounds[2])/2
        y = (bounds[1]+bounds[3])/2
        gc.collect()
    return True, [f_name, x, y, continent]


def run_sampler_chunk(chunk_data):
    """
    Function to process a chunk of points for sampling in parallel.
    """
    count = 0
    results = []
    for idx, row in chunk_data.iterrows():
        try:
            lon, lat = row.geometry.x, row.geometry.y
            index = row["rand_point_id"]
            continent = row.CONTINENT
            save_params = {
                "continent": "images",
                # "sample_id": start_idx + count,
                "sample_id": index,
                "output_dir": OUT_DIR
            }
            sample_found, metadata = patch_sampler(
                lon=lon,
                lat=lat,
                continent=continent,
                save_params=save_params
            )
            if sample_found:
                count += 1
                results.append(metadata)
        except Exception as e:
            print(f"Error at ID {idx}: {e}")
            if REPORT_BAD:
                with open(TXT_FILE, "a") as f:  # Append mode ensures data isn't overwritten
                    f.write(f"Error {save_params['sample_id']}\n")
            continue
    gc.collect()
    return count, results


def run_sampler_parallel(data, index_range):
    """
    Runs the sampler in parallel using multiprocessing.
    """
    # Define the number of parallel processes (cores) to use.
    num_processes = 12  # Adjust this if you want to use fewer cores
    print(f'Number of processes: {num_processes}')
    # Split the data into chunks for each process
    chunk_size = len(data) // num_processes
    print(f'Chunk size: {chunk_size}')
    chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    with ProcessPoolExecutor(max_workers=12) as executor:
        # Use unpacking with map
        results = list(executor.map(run_sampler_chunk, chunks))

    # Sum the counts from each process
    total_samples = sum([r[0] for r in results]) # count on 0th index
    results = [r[1] for r in results]
    results = list(itertools.chain.from_iterable(results))
    # metadata = list(itertools.chain(list1, list2, list3))
    df = pd.DataFrame(results, columns=["fn","lon", "lat", "Continent"])
    df.to_csv(f'{OUT_DIR}/index_{index_range}.csv', index=False)
    print(f"Total samples collected: {total_samples}")
    return total_samples

# Define parameters as before
# lonlat_samples_pth = "/jet/home/muhawena/data_scratch/muhawena/datasets/satCLIP_data/biased_AS/lon_lat_samples/asia_70k_lonlat.csv"

import sys

if __name__ == '__main__':
    arguments = sys.argv[1:]  # Exclude the script name
    index_range = arguments[0]
    print(f'Downloading range {index_range}')
    OUT_DIR = "/scratch/akaur64/GeoSSL/satclip/stra_con"
    REPORT_BAD = False
    # /{index_range}" # removing to get all images at one place
    DATE_RANGE = "2023-01-01/2023-12-01"
    PATCH_SIZE = (256, 256)
    CLOUD_COVER = 20
    TXT_FILE = f"{OUT_DIR}/bad_files_{index_range}.txt"
    LOG_FILE = f"{OUT_DIR}/log.txt"


    data = gpd.read_file(f"./split_geojsons/{index_range}.geojson") # this is a csv or a geojson for sampled points
    # data = data.iloc[int(index_range.split('-')[0]):int(index_range.split('-')[1])]
    #### CONTINUATION LOGIC IF NEEDED #### NOT NEEDED WITH 2000 SIZE CHUNKS ####
    
    # try:
    #     files = os.listdir(f'{OUT_DIR}/images')
    #     indexes = [int(w.split('.')[0][6:]) for w in files]
    #     try:
    #         with open(TXT_FILE, "r") as f:
    #             content = f.read()  # Read the entire file as a string
    #         sentences = content.split("\n")
    #         bad_indexes = [int(sen.split()[1]) for sen in sentences[:-1]]
    #         indexes = indexes + bad_indexes
    #         indexes = [index-int(index_range.split('-')[0]) for index in indexes]
    #     except FileNotFoundError as e:
    #         print("intial bad indexes not found")
    #     data = data.drop(data.index[indexes])
    # except FileNotFoundError as e:
    #     print("initial images not found")
    #     pass

    #### CONTINUATION LOGIC IF NEEDED #### NOT NEEDED WITH 2000 SIZE CHUNKS ####
    # Call the run_sampler_parallel function
    total_samples = run_sampler_parallel(data, index_range)

    print("Number of samples downloaded: ", total_samples)

# find /scratch/akaur64/GeoSSL/satclip/stra_con/images/patch_1* | parallel --progress -j 60 mv {} /data/hkerner/geossl/SatCLIP_pretraining_data/aman_downloaded_small/images
