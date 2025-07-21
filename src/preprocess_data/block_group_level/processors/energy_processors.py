import pandas as pd
import geopandas as gpd
import os
import shapely as sp
import shapely.wkt as wkt

from src.preprocess_data.block_group_level.utils import (
    parse_geo_codes,
)

from src.GLOBAL import FIPS_MAPPING_DF

METER2_TO_KM2 = 1e-6  # Conversion factor from square meters to square kilometers

METER2_TO_MI2 = 1e-6 * 0.386102  # Conversion factor from square meters to square miles

def _get_GEOID_from_point(geographic_point, block_group_bb):
    # Convert the geographic point to a Shapely Point object and check for intersection
    intersect = block_group_bb[block_group_bb.intersects(wkt.loads(geographic_point))]
    # Handle cases with no or multiple intersections
    if len(intersect) == 0:
        print(f"No intersection found for point: {geographic_point}")
        return (None, None, None, None, None)
    elif len(intersect) > 1:
        print(f"More than one intersection found for point: {geographic_point}")
        return (None, None, None, None, None)

    # Return the relevant details from the intersections
    geoid_information = (
        intersect.iloc[0]["GEOID"],
        intersect.iloc[0]["STATEFP"],
        intersect.iloc[0]["COUNTYFP"],
        intersect.iloc[0]["TRACTCE"],
        intersect.iloc[0]["BLKGRPCE"],
        intersect.iloc[0]["geometry"],
    )
    return geoid_information


def process_raw_solar_data(
    data_file_path: str, raw_bounding_box_filepath: str
) -> pd.DataFrame:
    raw_solar_data = pd.read_csv(data_file_path)
    raw_bounding_box = gpd.read_file(os.path.join(raw_bounding_box_filepath, "cb_2023_us_bg_500k.shp"), dtype={'GEOID': str, 'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str, 'BLKGRPCE': str})

    raw_solar_data[["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "geometry"]] = raw_solar_data["WKT"].apply(
        lambda x: pd.Series(_get_GEOID_from_point(x, raw_bounding_box))
    )

    # convert raw_solar_data to a geoDataFrame
    raw_solar_data = gpd.GeoDataFrame(
        raw_solar_data, geometry="geometry", crs=raw_bounding_box.crs
    ).to_crs(epsg=5070)
    # Filter out to relevant columns
    cols_of_interest = [
        "GEOID",
        "STATEFP",
        "COUNTYFP",
        "TRACTCE",
        "BLKGRPCE",
        "solar_mw",
        "geometry",
    ]
    filtered_solar_data = raw_solar_data[cols_of_interest]
    
    # calculate area in km2 and mi2
    filtered_solar_data["area km2"] = filtered_solar_data["geometry"].apply(lambda x: x.area * METER2_TO_KM2)
    filtered_solar_data["area mi2"] = filtered_solar_data["geometry"].apply(lambda x: x.area * METER2_TO_MI2)

    # aggregate solar data by GEOID and get the mean, sum, and count
    solar_aggregated = filtered_solar_data.groupby("GEOID").agg(
        {
            "solar_mw": ["mean", "sum", "count"],
            "area km2": "first",
            "area mi2": "first",
        }
    ).reset_index()
    
    # flatten column names
    solar_aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] for col in solar_aggregated.columns.values]
    solar_aggregated.rename(columns={
        'area km2_first': 'area_km2',
        'area mi2_first': 'area_mi2'
    }, inplace=True)
    
    # add 'US' to the front of the GEOID to keep consistent with what `parse_geo_codes` expects
    solar_aggregated["GEOID"] = "US" + solar_aggregated["GEOID"]
    
    # extract the state, county, tract, and block group codes from GEOID
    solar_aggregated[["STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE"]] = solar_aggregated["GEOID"].apply(
        lambda x: pd.Series(parse_geo_codes(x))
    )
    

    # merge with FIPS mapping DataFrame to add state and county names
    solar_aggregated = solar_aggregated.merge(
        FIPS_MAPPING_DF,
        left_on=["STATEFP", "COUNTYFP"],
        right_on=["FIPS State", "FIPS County"],
        how="inner"
    ).drop(columns=["FIPS State", "FIPS County", "GEOID"])
    
    # get the per km2 and per mi2 values for all the solar data
    solar_aggregated = _get_per_area_values(solar_aggregated, "solar_mw_mean")
    solar_aggregated = _get_per_area_values(solar_aggregated, "solar_mw_sum")
    solar_aggregated = _get_per_area_values(solar_aggregated, "solar_mw_count")
    
    return solar_aggregated
    
def _get_per_area_values(solar_aggregated: pd.DataFrame, 
                         metric: str,
) -> pd.DataFrame:
    """
    Calculate the per area values for the solar data.
    
    Args:
        solar_aggregated (pd.DataFrame): The aggregated solar data.
        metric (str): The metric to calculate per area values for ('solar_mw_mean', 'solar_mw_sum', 'solar_mw_count').
        
    Returns:
        pd.DataFrame: The solar data with per area values added.
    """
    solar_aggregated[f"{metric}_per_km2"] = solar_aggregated[metric] / solar_aggregated["area_km2"]
    solar_aggregated[f"{metric}_per_mi2"] = solar_aggregated[metric] / solar_aggregated["area_mi2"]
    
    return solar_aggregated
    

    
