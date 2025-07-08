import pandas as pd
import geopandas as gpd
from ..GLOBAL import FIPS_MAPPING_DF
import os

METER2_TO_KM2 = 1e-6  # Conversion factor from square meters to square kilometers

METER2_TO_MI2 = 1e-6 * 0.386102  # Conversion factor from square meters to square miles

def process_raw_county_bounding_box(filepath: str) -> pd.DataFrame:
    """
    Reads a shapefile containing county bounding box data and converts it to a DataFrame.

    Args:
        filepath (str): Path to the shapefile directory containing bounding box data.

    Returns:
        pd.DataFrame: A DataFrame with county bounding box geometries.
    """
    raw_county_bounding_box = gpd.read_file(os.path.join(filepath, "cb_2018_us_county_500k.shp"), dtype={'GEOID': str, 'STATEFP': str, 'COUNTYFP': str})
    
    # merge with FIPS mapping DataFrame to add state and county names
    raw_county_bounding_box = raw_county_bounding_box.merge(
        FIPS_MAPPING_DF,
        left_on=['STATEFP', 'COUNTYFP'],
        right_on=["FIPS State", "FIPS County"],
        how="inner"
    ).drop(columns=["FIPS State", "FIPS County"])
    
    # Use the geometry column to calculate the bounding box using epsg 5070
    raw_county_bounding_box['area km2'] = raw_county_bounding_box['geometry'].to_crs(epsg=5070).apply(lambda geom: geom.area * METER2_TO_KM2)   
    raw_county_bounding_box['area mi2'] = raw_county_bounding_box['geometry'].to_crs(epsg=5070).apply(lambda geom: geom.area * METER2_TO_MI2)
    
    # Cleaining up the DataFrame to include only relevant columns
    county_bounding_box_final = raw_county_bounding_box[[
        "GEOID", "STATEFP", "COUNTYFP", "State", "County Name", "area km2", "area mi2"
    ]].copy()
    
    #################################
    # Removing duplicates and some hand picked counties with data errors 
    #################################
    # Remove duplicates by keeping the first occurrence
    county_bounding_box_final = county_bounding_box_final.drop_duplicates(subset=["State", "County Name"], keep='first') # type: ignore

    # INDEX_TO_REMOVE = [1703, 1318, 2646, 2617, 2626, 2605]
    # county_bounding_box_final = county_bounding_box_final.drop(index=INDEX_TO_REMOVE)
    
    return county_bounding_box_final
    
    