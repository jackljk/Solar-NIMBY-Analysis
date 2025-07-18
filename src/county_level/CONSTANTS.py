RAW_DATA_DIRECTORY = "../../data" # Add the path to the raw data folder here


data_file_mappings_county_raw = {
    "Wind": "../projects/wind/ez_gis.plant_power_eia_v8_wind.shp",
    "Solar": "../projects/solar/solar_raw.csv",
    "solar_roof": "../projects/solar/solar_roof_raw.csv",
    "GDP": "social factors/gdp_raw.csv",
    "education": "social factors/education_raw.csv",
    "private_schools": "social factors/private_school_raw.csv",
    "DEC_race": "social factors/race_dec_raw.csv",
    "ACS_race": "social factors/race_acs_raw.csv",
    "election": "social factors/election_raw.csv",
    "income": "social factors/income_raw.csv",
    "unemployment": "social factors/unemployment_raw.csv",
    'population_data': 'social factors/population_raw.csv',
    "NREL_Electric": "electric price/NREL_raw.csv",
    "EIA_Electric": "electric price/EIA_raw.csv",
    "Rural_Urban": "electric price/rural_urban_raw.csv",
}


data_file_mappings_extras = {
    "FIPS": "US_FIPS_Codes.csv",
}

data_file_mappings_county_clean = {
    "bounding_boxes": "county_bounding_boxes_full.csv",
}


def get_file_path(file_name):
    if file_name in data_file_mappings_county_raw:
        return f"{RAW_DATA_DIRECTORY}/county_raw/{data_file_mappings_county_raw[file_name]}"
    elif file_name in data_file_mappings_extras:
        return f"{RAW_DATA_DIRECTORY}/extras/{data_file_mappings_extras[file_name]}"
    elif file_name in data_file_mappings_county_clean:
        return f"{RAW_DATA_DIRECTORY}/county_clean/{data_file_mappings_county_clean[file_name]}"
