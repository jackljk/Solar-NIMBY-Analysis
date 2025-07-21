import pandas as pd
from typing import Optional
from ..utils import (
    add_geographic_identifiers, convert_columns_to_numeric,
    clean_and_validate_data, select_final_columns
)


def process_raw_election_data(data_file_path: str, fips_mapping_path: Optional[str] = None) -> pd.DataFrame:
    """
    Processes raw election data from a CSV file for block group level.

    Args:
        data_file_path (str): Path to the raw election data CSV file
        fips_mapping_path (str, optional): Path to FIPS mapping file. If None, assumes FIPS data is available globally.

    Returns:
        pd.DataFrame: Processed DataFrame with election data and geographic identifiers
    """
    # Read the raw election data
    df = pd.read_csv(data_file_path, dtype={
        'geoid': str,
        'statefp': str,
        'countyfp': str,
        'tractce': str,
        'blkgrpce': str,
    })
    
    # Load FIPS mapping if path provided
    if fips_mapping_path:
        fips_df = pd.read_csv(fips_mapping_path)
    else:
        # Try to import from global context (assuming it's available)
        try:
            from src.GLOBAL import FIPS_MAPPING_DF as fips_df
        except ImportError:
            raise ValueError("FIPS mapping data not available. Please provide fips_mapping_path parameter.")
    
    # Check if required columns exist
    required_cols = ['geoid', 'statefp', 'countyfp', 'votes_total', 'votes_rep', 'votes_dem']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in election data: {missing_cols}")
    
    # Merge with FIPS data to get state and county names
    if 'FIPS State' in fips_df.columns and 'FIPS County' in fips_df.columns:
        merged = pd.merge(
            df, 
            fips_df, 
            left_on=['statefp', 'countyfp'], 
            right_on=['FIPS State', 'FIPS County'], 
            how='left'
        ).drop(columns=['FIPS State', 'FIPS County'], errors='ignore')
    else:
        merged = df.copy()
        print("Warning: FIPS mapping columns not found, proceeding without state/county names")
    
    # Calculate vote percentages
    merged['vote_rep_percent'] = merged['votes_rep'] / merged['votes_total'].replace(0, pd.NA)
    merged['vote_dem_percent'] = merged['votes_dem'] / merged['votes_total'].replace(0, pd.NA)
    
    # Rename columns for consistency
    rename_dict = {
        'geoid': 'GEOID',
        'votes_total': 'Total Votes',
        'votes_rep': 'Republican Votes', 
        'votes_dem': 'Democratic Votes',
        'vote_rep_percent': 'Republican Vote Percentage',
        'vote_dem_percent': 'Democratic Vote Percentage'
    }
    merged.columns = [rename_dict.get(col, col) for col in merged.columns]
    
    # Add geographic identifiers using GEOID if available
    if 'GEOID' in merged.columns:
        # Parse GEOID to extract state, county, tract, block group codes
        merged['GEOID'] = 'US' + merged['GEOID'].astype(str)
        merged = add_geographic_identifiers(merged, 'GEOID', slice_start=0)
        merged = merged.drop(columns=['GEOID'], errors='ignore')
    
    # Convert numeric columns to proper numeric types
    numeric_columns = [
        'Total Votes',
        'Republican Votes',
        'Democratic Votes', 
        'Republican Vote Percentage',
        'Democratic Vote Percentage'
    ]
    merged = convert_columns_to_numeric(merged, numeric_columns)
    
    # Remove unnecessary columns
    cols_to_remove = ['intptlat', 'intptlon', 'statefp', 'countyfp']
    merged = merged.drop(columns=cols_to_remove, errors='ignore')
    
    # Select final columns in desired order
    final_columns = [
        'GEOID',
        'State',
        'County',
        'Tract', 
        'Block Group',
        'State Name',
        'County Name',
        'Total Votes',
        'Republican Votes',
        'Democratic Votes',
        'Republican Vote Percentage',
        'Democratic Vote Percentage'
    ]
    
    merged = select_final_columns(merged, final_columns)
    
    # Clean and validate data
    merged = clean_and_validate_data(merged, "election data")
    
    return merged