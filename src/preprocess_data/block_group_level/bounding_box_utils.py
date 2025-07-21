import pandas as pd
import geopandas as gpd
from src.GLOBAL import FIPS_MAPPING_DF
import os

METER2_TO_KM2 = 1e-6  # Conversion factor from square meters to square kilometers

METER2_TO_MI2 = 1e-6 * 0.386102  # Conversion factor from square meters to square miles


def process_raw_block_group_bounding_box(filepath: str) -> pd.DataFrame:
    """
    Reads a shapefile containing bounding box data and converts it to a DataFrame.

    Args:
        filepath (str): Path to the shapefile directory containing bounding box data.

    Returns:
        pd.DataFrame: A DataFrame with bounding box geometries.
    """
    raw_block_group_bounding_box = gpd.read_file(os.path.join(filepath, "cb_2023_us_bg_500k.shp"), dtype={'GEOID': str, 'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str, 'BLKGRPCE': str})
    
    # Merging with FIPS mapping DataFrame to add state and county names
    raw_block_group_bounding_box = raw_block_group_bounding_box.merge(
        FIPS_MAPPING_DF,
        left_on=['STATEFP', 'COUNTYFP'],
        right_on=["FIPS State", "FIPS County"],
        how="inner"
    ).drop(columns=["FIPS State", "FIPS County"])

    # Use the geometry column to calculate the bounding box using epsg 5070
    raw_block_group_bounding_box['area km2'] = raw_block_group_bounding_box['geometry'].to_crs(epsg=5070).apply(lambda geom: geom.area * METER2_TO_KM2)

    raw_block_group_bounding_box['area mi2'] = raw_block_group_bounding_box['geometry'].to_crs(epsg=5070).apply(lambda geom: geom.area * METER2_TO_MI2)
    
    block_group_bounding_box_final = raw_block_group_bounding_box[[
        "GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "State", "County Name", "area km2", "area mi2"
    ]].copy()
    
    # Ensure we're returning a DataFrame
    if isinstance(block_group_bounding_box_final, pd.Series):
        block_group_bounding_box_final = block_group_bounding_box_final.to_frame().T
    
    return block_group_bounding_box_final