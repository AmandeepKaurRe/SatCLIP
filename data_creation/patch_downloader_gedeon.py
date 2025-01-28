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

TXT_FILE = f"/scratch/akaur64/GeoSSL/satclip/bad_files_20000.txt"
LOG_FILE = f"/scratch/akaur64/GeoSSL/satclip/log_20000.txt"

def log_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    with open(LOG_FILE, "a") as f:  # Append mode ensures data isn't overwritten
        f.write(f"{mem:.2f}, {pid}, {datetime.now()}\n")

index_range = '40_000-60_000'
output_folder = f"/scratch/akaur64/GeoSSL/satclip/{index_range}"
txt_file = f'{output_folder}/bad_indexes.txt'


# This function will take lon/lat, and write to file the xarry patch sampled
def patch_sampler(lon, lat, continent, time_of_interest, cloud_cover=10,patch_size=(256,256), save_params=None):
    # try:
    area_of_interest = {"type": "Point", "coordinates": [lon, lat]} # example for rwanda point: lon=29.9309, lat = -1.9154

    search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": cloud_cover}},
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

    x = np.random.randint(0, width - patch_size[0])
    y = np.random.randint(0, height - patch_size[1])

    patch = stack[0, :, y : y + patch_size[1], x : x + patch_size[0]].compute()
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
        # print("Found collection is too cloudy")
        with open(TXT_FILE, "a") as f:  # Append mode ensures data isn't overwritten
            f.write(f"empty {save_params['sample_id']}\n")
        return False, []
    else:
        raster_path = patch_path
        patch.values = np.nan_to_num(patch.values)
        patch.values = patch.values.astype(np.uint16)  # Ensure correct data type
        patch.rio.to_raster(raster_path=patch_path, driver="COG", dtype=np.uint16, compress="LZW", nodata=0)
        bounds = stackstac.array_bounds(patch, to_epsg=4326)
        x = (bounds[0]+bounds[2])/2
        y = (bounds[1]+bounds[3])/2
    return True, [f_name, x, y, continent]
    # except Exception as e:
    #     print(f"Error at lon-lat {lon},{lat}: {e}")
    #     return False, []

def run_sampler_chunk(chunk_data, start_idx, output_folder, samples_collection_dates, patch_size, cloud_cover):
    """
    Function to process a chunk of points for sampling in parallel.
    """
    count = 0
    results = []
    for idx, row in chunk_data.iterrows():
        try:
            log_memory_usage()
            lon, lat = row.geometry.x, row.geometry.y
            continent = row.CONTINENT
            save_params = {
                "continent": "images",
                "sample_id": start_idx + count,
                "output_dir": output_folder
            }
            sample_found, metadata = patch_sampler(
                lon=lon,
                lat=lat,
                continent=continent,
                time_of_interest=samples_collection_dates,
                cloud_cover=cloud_cover,
                patch_size=patch_size,
                save_params=save_params
            )
            if sample_found:
                count += 1
                results.append(metadata)
        except Exception as e:
            print(f"Error at ID {idx}: {e}")
            with open(TXT_FILE, "a") as f:  # Append mode ensures data isn't overwritten
                f.write(f"Error {save_params['sample_id']}\n")
            continue
    return count, results


def run_sampler_parallel(data, output_folder, samples_collection_dates, patch_size=(256, 256), cloud_cover=20):
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
    # Create a pool of workers and process each chunk in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            run_sampler_chunk,
            [(chunk, i * chunk_size, output_folder, samples_collection_dates, patch_size, cloud_cover) for i, chunk in enumerate(chunks)]
        )

    # Sum the counts from each process
    total_samples = sum([r[0] for r in results]) # count on 0th index
    results = [r[1] for r in results]
    results = list(itertools.chain.from_iterable(results))
    # metadata = list(itertools.chain(list1, list2, list3))
    df = pd.DataFrame(results, columns=["fn","lon", "lat", "Continent"])
    df.to_csv(f'{output_folder}/images/index.csv', index=False)
    print(f"Total samples collected: {total_samples}")
    return total_samples

# Define parameters as before
idx = 0
count = 0

samples_collection_dates = "2023-01-01/2023-12-01"
patch_size = (256, 256)
cloud_cover = 20

# Load the data
# df = pd.read_csv(lonlat_samples_pth)
data = gpd.read_file(f"./geojsons/{index_range}.geojson") # this is a csv or a geojson for sampled points

try:
    files = os.listdir(f'{output_folder}/images')
    indexes = [int(w.split('.')[0][6:]) for w in files]
    try:
        with open(txt_file, "r") as f:
            content = f.read()  # Read the entire file as a string
        sentences = content.split("\n")
        bad_indexes = [sen.split(' ')[1] for sen in sentences[:-1]]
        data = data.drop(data.index[bad_indexes])
    except FileNotFoundError as e:
        print("intial bad indexes not found")
    data = data.drop(data.index[indexes])
except FileNotFoundError as e:
    print("initial images not found")
    pass

# Call the run_sampler_parallel function
total_samples = run_sampler_parallel(
    data=data,
    output_folder=output_folder,
    samples_collection_dates=samples_collection_dates,
    patch_size=patch_size,
    cloud_cover=cloud_cover
)

print("Number of samples downloaded: ", total_samples)