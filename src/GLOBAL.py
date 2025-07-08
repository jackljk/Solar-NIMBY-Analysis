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
    dtype={"FIPS State": str, "FIPS County": str}
)

