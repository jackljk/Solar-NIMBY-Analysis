import os
from typing import Union

from src.GLOBAL import ROOT_DIR


######################################################
# Directory paths for raw suitability scores tif files
######################################################
SUITABILITY_SCORES_DIR = os.path.join(ROOT_DIR, "data", "raw_suitability_data")

# variables file names
GHI = "GHI-09188ce2.tif"
PROTECTED_LAND = "Protected_Land-5745a356.tif"
HABITAT = "Habitat-32079c87.tif"
SLOPE = "slope_only-2c1658fa.tif"
POPL_DENS = "Popl_Density-714f0a64.tif"
SUBSTATION = "distance_to_substation_only-f02c9129.tif"
LAND_COVER = "Land_Cover-8a2691e6.tif"

variables = [GHI, PROTECTED_LAND, HABITAT, SLOPE, POPL_DENS, SUBSTATION, LAND_COVER]
variables_names = [
    "GHI",
    "Protected_Land",
    "Habitat",
    "Slope",
    "Population_Density",
    "Distance_to_Substation",
    "Land_Cover",
]

SUITABILITY_SCORES_FILES = {}
for var, name in zip(variables, variables_names):
    SUITABILITY_SCORES_FILES[name] = os.path.join(SUITABILITY_SCORES_DIR, var)

######################################################
# Utility Functions
######################################################
import rasterio
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import Point
from scipy.ndimage import distance_transform_edt
from typing import Union

from src.GLOBAL import FIPS_MAPPING_DF


def calculate_zonal_stats(tif_path, geodataframe, nodata_value):
    with rasterio.open(tif_path) as src:
        affine = src.transform
        array = src.read(1)  # Read the first band
        array = np.where(
            np.isnan(array), nodata_value, array
        )  # Replace NaNs with nodata_value
        # # Debugging: Check raster data and affine transformation
        # print(f"Raster data shape: {array.shape}")
        # print(f"Affine transformation: {affine}")
        # Check the CRS of the raster
        raster_crs = src.crs
        # print(f"Raster CRS: {raster_crs}")
        if geodataframe.crs != raster_crs:
            geodataframe = geodataframe.to_crs(raster_crs)

        # filter out values that are less than 100
        array = np.where(array > 101, nodata_value, array)

    # Calculate zonal statistics
    stats = zonal_stats(
        geodataframe,
        array,
        affine=affine,
        stats="mean",
        nodata=nodata_value,
        all_touched=True,
    )
    # Extract mean values and add to GeoDataFrame
    mean_values = [stat["mean"] for stat in stats]
    return mean_values


def process_tif_files(
    tif_filepaths, bounding_box, nodata_value=-9999, block_group=False
):
    bounding_box_copy = bounding_box.copy()

    # Initialize results DataFrame
    results = pd.DataFrame(index=bounding_box_copy.index, columns=variables_names)

    for tif_path, col_name in zip(tif_filepaths, variables_names):
        # print(f"Processing {tif_path} for {col_name}")

        # Calculate mean values using zonal stats
        mean_values = calculate_zonal_stats(tif_path, bounding_box_copy, nodata_value)

        # Update results DataFrame
        results[col_name] = mean_values

    # Add county and state information
    results["County Name"] = bounding_box["County Name"]
    results["State"] = bounding_box["State"]
    if block_group:
        results["GEOID"] = bounding_box["GEOID"]
        results["TRACTCE"] = bounding_box["TRACTCE"]
        results["BLKGRPCE"] = bounding_box["BLKGRPCE"]

    return results


def handle_Connecticut_county_mapping(
    dataframe: Union[pd.DataFrame, gpd.GeoDataFrame],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Handle the special case of Connecticut county mapping.

    Args:
        dataframe (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame or GeoDataFrame.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: The modified DataFrame with corrected county names.
    """
    mapper = {
        "110": "Hartford",
        "190": "Fairfield",
        "170": "Litchfield",
        "140": "Middlefield",
        "120": "New Haven",
        "130": "Tolland",
        "160": "Windham",
        "180": "New London",
        "150": "New London",
    }

    def fix_data_Connecticut(series):

        if series["GEOID"][:2] == "09":
            series["State"] = "Connecticut"
            series["County Name"] = mapper[series["GEOID"][2:5]]
            series["TRACTCE"] = series["GEOID"][5:11]
            series["BLKGRPCE"] = series["GEOID"][11:]

        return series
    
    dataframe = dataframe.copy() # make a copy to avoid modifying the original dataframe
    dataframe = dataframe.apply(fix_data_Connecticut, axis=1)
    
    return dataframe

def handle_code_matching_error(
    dataframe: Union[pd.DataFrame, gpd.GeoDataFrame],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Handle the special case of code matching errors in the DataFrame.

    Args:
        dataframe (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame or GeoDataFrame.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: The modified DataFrame with corrected codes.
    """
    state_dict = FIPS_MAPPING_DF.set_index('FIPS State')['State'].to_dict()

    # County dict for mapping but requires to match the FIPS State as well
    county_dict = FIPS_MAPPING_DF.set_index(['FIPS State', 'FIPS County'])['County Name'].to_dict()
    def fix_tract(series):
        if series['GEOID'] != np.nan:
            if series['GEOID'][:2] == '02':
                series['State'] = 'Alaska'
            elif series['GEOID'][:2] == '15':
                series['State'] = 'Hawaii'
            else:
                try:
                    series['State'] = state_dict[series['GEOID'][:2]]
                except KeyError:
                    series['State'] = np.nan
            series['TRACTCE'] = series['GEOID'][5:11]
            series['BLKGRPCE'] = series['GEOID'][11:]
            try:
                series['County Name'] = county_dict[(series['GEOID'][:2], series['GEOID'][2:5])]
            except KeyError:
                series['County Name'] = np.nan
            
        return series
    
    dataframe = dataframe.copy()  # make a copy to avoid modifying the original dataframe
    dataframe = dataframe.apply(fix_tract, axis=1)  
    
    return dataframe


#############################################
# Utility Functions for Project Level suitability scores
#############################################
def get_average_value(array, row, col, window_size=3, nodata_value=255):
    """
    Calculate the average value of pixels in a square window around a specified row and column,
    ignoring nodata values.

    Args:
        array: 2D NumPy array of raster data.
        row: Row index of the point in the raster.
        col: Column index of the point in the raster.
        window_size: Half the size of the square window (e.g., 3 means a 7x7 window).
        nodata_value: Value representing nodata in the raster.

    Returns:
        Average value of the valid pixels in the window. If all pixels are nodata, return None.
    """
    # Define the window boundaries
    row_min = max(0, row - window_size)
    row_max = min(array.shape[0], row + window_size + 1)
    col_min = max(0, col - window_size)
    col_max = min(array.shape[1], col + window_size + 1)

    # Extract the window
    window = array[row_min:row_max, col_min:col_max]

    # Mask out nodata values
    valid_values = window[window != nodata_value]

    # If no valid values are present, return None
    if valid_values.size == 0:
        return None

    # Calculate and return the mean of valid values
    return valid_values.mean()


def get_closest_edge_value(array, row, col, nodata_value=255):
    """
    Finds the closest raster value for a point outside the raster bounds.

    Args:
        array: 2D NumPy array of raster data.
        row: Row index of the point in the raster.
        col: Column index of the point in the raster.
        nodata_value: Value representing nodata in the raster.

    Returns:
        Cloest valid value indices
    """
    # Clamp the row and col to raster bounds
    row = min(max(0, row), array.shape[0] - 1)
    col = min(max(0, col), array.shape[1] - 1)

    # If the value is nodata, find the nearest valid value
    if array[row, col] == nodata_value:
        valid_mask = array != nodata_value
        distances, indices = distance_transform_edt(valid_mask, return_indices=True)
        closest_index = indices[:, row, col]
        return array[tuple(closest_index)], distances[row, col]

    return array[row, col], 0


def calculate_point_values_with_average(tif_path, geodataframe, window_size=3, nodata_value=255):
    """
    Extracts the average raster values around point locations using a square window,
    handling out-of-bounds cases.
    """
    with rasterio.open(tif_path) as src:
        affine = src.transform
        raster_crs = src.crs
        array = src.read(1)
        height, width = array.shape

        # Ensure CRS alignment
        geodataframe = geodataframe.to_crs(raster_crs)
        # array = np.where(np.isnan(array), nodata_value, array)
        

        values = []
        distances = []
        for i, geom in enumerate(geodataframe.geometry):
            if geom.has_z:
                geom = Point(geom.x, geom.y)

            # Map point to raster indices
            col, row = ~affine * (geom.x, geom.y)
            row, col = int(row), int(col)

            # Check if indices are within bounds
            if 0 <= row < height and 0 <= col < width:
                # Get the average value around the point
                value = get_average_value(array, row, col, window_size, nodata_value)
            else:
                # Return nodata_value if out of bounds
                value = np.nan
                print(f"Point {i} is out of bounds")
                
            values.append(value)
            # Debugging output
            if i % 100 == 0:
                print(f"Processed {i}/{len(geodataframe.geometry)} points")
        return values, distances


def process_tif_files_with_average(tif_filepaths, geodataframe, window_size=50, nodata_value=255, get_distance=False):
    """
    Process multiple TIFF files and extract average values around point locations using a square window,
    handling out-of-bounds cases.
    """
    # Convert WKT Point Z strings to geometries if necessary
    if isinstance(geodataframe.geometry.iloc[0], str):
        geodataframe["geometry"] = geodataframe["geometry"].apply(loads)
    geodataframe = gpd.GeoDataFrame(geodataframe, geometry="geometry", crs="EPSG:4326")  # Adjust CRS as needed

    # Initialize results DataFrame
    results = pd.DataFrame(index=geodataframe.index, columns=variables_names)

    for tif_path, col_name in zip(tif_filepaths, variables_names):
        print(f"Processing {tif_path} for {col_name}")

        # Extract average values from the raster around point locations
        values, distances = calculate_point_values_with_average(tif_path, geodataframe, window_size, nodata_value)

        # Update results DataFrame
        results[col_name] = values
        if get_distance:
            results[f"{col_name}_distance"] = distances

    # Add geometry back to the results DataFrame
    results["geometry"] = geodataframe["geometry"]
    # Add the wattage column back to the results DataFrame
    results["Wattage"] = geodataframe["total_mw"]

    return results


def get_GEOID_from_point(point, block_group_bb):
    """
    Get GEOID and related geographic identifiers from a point geometry.
    
    Args:
        point: Shapely Point geometry
        block_group_bb: GeoDataFrame containing block group boundaries
        
    Returns:
        tuple: (GEOID, STATEFP, COUNTYFP, TRACTCE, BLKGRPCE) or (None, None, None, None, None)
    """
    intersect = block_group_bb[block_group_bb.intersects(point)]
    
    # Handle cases with no or multiple intersections
    if len(intersect) == 0:
        print(f"No intersection found for point: {point}")
        return (None, None, None, None, None)
    elif len(intersect) > 1:
        print(f"Multiple intersections found for point: {point}")
        return (None, None, None, None, None)
    
    # Return the relevant details from the intersection
    row = intersect.iloc[0]
    return (row['GEOID'], row['STATEFP'], row['COUNTYFP'], row['TRACTCE'], row['BLKGRPCE'])