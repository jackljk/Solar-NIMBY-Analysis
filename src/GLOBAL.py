"""
This file contains global variables and constants used throughout the application.
"""

import os
import pandas as pd

# Define the root directory of the project (parent of src)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Root directory set to: {ROOT_DIR}")

STATES_TO_DROP = [
    "American Samoa",
    "Puerto Rico",
    "Alaska",
    "Hawaii",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
    "Guam",
]

# DataFrame containing FIPS state and county code mappings
FIPS_MAPPING_DF = pd.read_csv(
    os.path.join(ROOT_DIR, "data", "mappings", "US_FIPS_Codes.csv"),
    dtype={"FIPS State": str, "FIPS County": str},
)

EIA_FIPS_MAPPING_DF = pd.read_csv(
    os.path.join(ROOT_DIR, "data", "mappings", "EIA_Codes.csv"),
    dtype={"utility_id_eia": str, "county_id_fips": str}
)[["county_id_fips", "utility_id_eia"]]


##########################################################
# Constants for energy data file paths (Global Data)
##########################################################

ENERGY_DATA_DIR = os.path.join(ROOT_DIR, "data", "electric_data")

##########################################################
# Constants for file paths (County Level Data)
##########################################################
COUNTY_LEVEL_DATA_DIR = os.path.join(ROOT_DIR, "data", "social_factors", "county_level")

COUNTY_LEVEL_DEMOGRAPHIC_DATA_DIR = os.path.join(
    COUNTY_LEVEL_DATA_DIR, "demographic_data"
)

COUNTY_LEVEL_GEOGRAPHIC_DATA_DIR = os.path.join(
    COUNTY_LEVEL_DATA_DIR, "geographic_data"
)   

COUNTY_LEVEL_ECONOMIC_DATA_DIR = os.path.join(
    COUNTY_LEVEL_DATA_DIR, "economic_data"
)

COUNTY_LEVEL_POLITICAL_DATA_DIR = os.path.join(
    COUNTY_LEVEL_DATA_DIR, "political_data"
)

COUNTY_LEVEL_BOUNDING_BOXES_DIR = os.path.join(
    COUNTY_LEVEL_DATA_DIR, "bounding_boxes_raw"
)

COUNTY_LEVEL_DATA_FILES = {
    # Demographic Data
    "education": os.path.join(
        COUNTY_LEVEL_DEMOGRAPHIC_DATA_DIR, "education_raw.csv"
    ),
    "race": COUNTY_LEVEL_DEMOGRAPHIC_DATA_DIR, # For race we give the directory
    "unemployment": os.path.join(
        COUNTY_LEVEL_DEMOGRAPHIC_DATA_DIR, "unemployment_raw.csv"
    ),
    # Economic Data
    "income": os.path.join(
        COUNTY_LEVEL_ECONOMIC_DATA_DIR, "income_raw.csv"
    ),
    "electric_price_EIA": os.path.join(
        COUNTY_LEVEL_ECONOMIC_DATA_DIR, "electric_price_EIA_raw.csv"
    ),
    "electric_price_NREL": os.path.join(
        COUNTY_LEVEL_ECONOMIC_DATA_DIR, "electric_price_NREL_raw.csv"
    ),
    "gdp": os.path.join(
        COUNTY_LEVEL_ECONOMIC_DATA_DIR, "gdp_raw.csv"
    ),
    # Geographic Data
    "private_schools": os.path.join(
        COUNTY_LEVEL_GEOGRAPHIC_DATA_DIR, "private_school_raw.csv"
    ),
    "rural_urban": os.path.join(
        COUNTY_LEVEL_GEOGRAPHIC_DATA_DIR, "rural_urban_raw.csv"
    ),
    # political Data
    "election": os.path.join(
        COUNTY_LEVEL_POLITICAL_DATA_DIR, "election_raw.csv"
    ),
    # General Data
    "population_data": os.path.join(
        COUNTY_LEVEL_DATA_DIR, "population_raw.csv"
    ),
    "bounding_boxes": COUNTY_LEVEL_BOUNDING_BOXES_DIR,  # For bounding boxes we give the directory

    # Energy Data
    "solar": os.path.join(ENERGY_DATA_DIR, "solar", "solar_raw.csv"), 
    "solar_roof": os.path.join(ENERGY_DATA_DIR, "solar", "solar_roof_raw.csv"),
    "wind": os.path.join(ENERGY_DATA_DIR, "wind"),  # For wind we give the directory
}


##########################################################
# Constants for file paths (Block Group Level Data)
##########################################################
BLOCK_GROUP_LEVEL_DATA_DIR = os.path.join(ROOT_DIR, "data", "social_factors", "block_group_level")

BLOCK_GROUP_LEVEL_DEMOGRAPHIC_DATA_DIR = os.path.join(
    BLOCK_GROUP_LEVEL_DATA_DIR, "demographic_data"
)

BLOCK_GROUP_LEVEL_ECONOMIC_DATA_DIR = os.path.join(
    BLOCK_GROUP_LEVEL_DATA_DIR, "economic_data"
)

BLOCK_GROUP_LEVEL_POLITICAL_DATA_DIR = os.path.join(
    BLOCK_GROUP_LEVEL_DATA_DIR, "political_data"
)

BLOCK_GROUP_LEVEL_BOUNDING_BOXES_DIR = os.path.join(
    ROOT_DIR, "data", "bounding_boxes", "block_group_bounding_box_raw"
)

BLOCK_GROUP_LEVEL_DATA_FILES = {
    # Demographic Data
    "education": os.path.join(
        BLOCK_GROUP_LEVEL_DEMOGRAPHIC_DATA_DIR, "education_raw.csv"
    ),
    "race": os.path.join(
        BLOCK_GROUP_LEVEL_DEMOGRAPHIC_DATA_DIR, "race_raw.csv"
    ),
    "unemployment": os.path.join(
        BLOCK_GROUP_LEVEL_DEMOGRAPHIC_DATA_DIR, "unemployment_raw.csv"
    ),
    # Economic Data
    "income": os.path.join(
        BLOCK_GROUP_LEVEL_ECONOMIC_DATA_DIR, "income_raw.csv"
    ),
    # Political Data
    "election": os.path.join(
        BLOCK_GROUP_LEVEL_POLITICAL_DATA_DIR, "election_raw.csv"
    ),
    # Bounding Box Data
    "bounding_boxes": BLOCK_GROUP_LEVEL_BOUNDING_BOXES_DIR,

    # Energy Data (shared with county level)
    "solar": os.path.join(ENERGY_DATA_DIR, "solar", "solar_raw_block_group.csv"),
}