# Example: Loading and filtering data for Dolo Ado in SE Ethiopia, one of the sites explored in our paper (4.17°N, 42.05°E)
import pandas as pd
import numpy as np
import rasterio
import math
from rasterio.transform import from_origin


# Load parquet data
df = pd.read_parquet('N03/N03E042/N03E042-post-processing.parquet')

# Apply recommended filters (for when aggregating over long time periods, not necessary for single event floods)
filtered_df = df[
    (df.dem_metric_2 < 10) &
    (df.soil_moisture_sca > 1) &
    (df.soil_moisture_zscore > 1) &
    (df.soil_moisture > 20) &
    (df.temp > 0) &
    (df.land_cover != 60) &
    (df.edge_false_positives == 0)
]

# when outputting as a raster, it's highly recommended to include the 80 meter buffer (4 pixels). See example code below
def df_to_geotiff(df, output_file, buffer_size=4, resolution=0.00018):
    """
    Convert a DataFrame of lat/lon points into a GeoTIFF with dilation.

    Args:
        df: DataFrame with columns ['lat', 'lon']
        output_file: Path to output GeoTIFF
        buffer_size: Number of pixels to dilate around each detection (4 → ~80 m)
        resolution: Pixel size in degrees (~0.00018 ≈ 20 m at equator)
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Nothing to rasterize.")

    lats = df['lat'].values
    lons = df['lon'].values

    # Compute raster bounds
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    width = int(math.ceil((max_lon - min_lon) / resolution)) + 1
    height = int(math.ceil((max_lat - min_lat) / resolution)) + 1

    transform = from_origin(min_lon, max_lat, resolution, resolution)

    # Initialize raster
    raster = np.zeros((height, width), dtype=np.uint8)

    # Burn points with dilation
    for lon, lat in zip(lons, lats):
        row = int((max_lat - lat) / resolution)
        col = int((lon - min_lon) / resolution)

        r_start, r_end = max(0, row - buffer_size), min(height, row + buffer_size + 1)
        c_start, c_end = max(0, col - buffer_size), min(width, col + buffer_size + 1)

        raster[r_start:r_end, c_start:c_end] = 1

    # Write GeoTIFF
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }

    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(raster, 1)

# creates geotiff with 20 meter resolution at equator (~0.00018 degrees), with an 4 pixel (80 meter) buffer. 
df_to_geotiff(filtered_df, 'flood_map.tif', buffer_size=4, resolution=0.00018)

# Load corresponding geotiff
with rasterio.open('N03/N03E042/N03E042-80m-buffer.tif') as src:
    flood_data = src.read(1)  # Read first band
