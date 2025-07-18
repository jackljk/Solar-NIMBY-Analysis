import pandas as pd
from src.county_level.utils import finalize_dataset


def process_raw_number_private_school_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw number of private school data from a CSV file.

    Args:
        data_file_path (str): Path to the raw number of private school data CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns.
    """
    # Read the CSV file with proper data types
    private_school_raw_data = pd.read_csv(data_file_path, dtype={"CNTY": str, "STFIP": str})
    
    # Select and copy relevant columns
    private_school_data = private_school_raw_data[["NAME", "STFIP", "CNTY"]].copy()
    
    # Extract county FIPS (last 3 digits) and rename state FIPS
    private_school_data["FIPS County"] = private_school_data["CNTY"].str[-3:]
    private_school_data = private_school_data.rename(columns={"STFIP": "FIPS State"})
    
    # Count private schools by county
    private_school_counts = (
        private_school_data
        .groupby(["FIPS State", "FIPS County"], as_index=False)
        .size()
        .rename(columns={"size": "No. of Private Schools"})
    )
    
    private_school_counts = finalize_dataset(private_school_counts)
    
    return private_school_counts

def process_raw_rural_urban_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw rural-urban classification data from a CSV file.

    Args:
        data_file_path (str): Path to the raw rural-urban classification data CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with rural and urban area percentages by county.
    """
    # Load data with proper data types for FIPS codes
    data = pd.read_csv(data_file_path, dtype={"STATE": str, "COUNTY": str})
    
    # Select relevant columns for rural-urban analysis and create a new DataFrame
    selected_data = data[['STATE', 'COUNTY', "ALAND_PCT_RUR", "ALAND_PCT_URB"]]
    rural_urban_data = pd.DataFrame({
        'FIPS State': selected_data['STATE'],
        'FIPS County': selected_data['COUNTY'],
        'ALAND_PCT_RUR': selected_data['ALAND_PCT_RUR'],
        'ALAND_PCT_URB': selected_data['ALAND_PCT_URB']
    })
    
    # Use finalize_dataset utility to merge with FIPS mapping and clean up
    merged_data = finalize_dataset(rural_urban_data)
    
    # Convert percentage strings to float values
    # Remove percentage signs and convert to decimal format
    if 'ALAND_PCT_RUR' in merged_data.columns:
        merged_data['ALAND_PCT_RUR'] = (
            merged_data['ALAND_PCT_RUR']
            .astype(str)
            .str.rstrip('%')
            .astype('float') / 100.0
        )
    
    if 'ALAND_PCT_URB' in merged_data.columns:
        merged_data['ALAND_PCT_URB'] = (
            merged_data['ALAND_PCT_URB']
            .astype(str)
            .str.rstrip('%')
            .astype('float') / 100.0
        )
    
    # Rename columns to more descriptive names
    column_mapping = {
        'ALAND_PCT_RUR': 'Rural Area Percentage',
        'ALAND_PCT_URB': 'Urban Area Percentage'
    }
    
    final_data = merged_data.rename(columns=column_mapping)
    
    # Ensure consistent column ordering
    column_order = ['State', 'County Name', 'Rural Area Percentage', 'Urban Area Percentage']
    final_columns = [col for col in column_order if col in final_data.columns]
    
    return final_data.reindex(columns=final_columns)
