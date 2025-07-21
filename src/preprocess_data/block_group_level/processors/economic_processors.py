import pandas as pd
from ..utils import (
    add_geographic_identifiers, add_area_names, convert_columns_to_numeric,
    clean_and_validate_data, select_final_columns
)


def process_raw_income_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw median household income data for block group level.
    
    Args:
        data_file_path (str): Path to income data file
    Returns:
        pd.DataFrame: Processed income data
    """
    # Read the raw income data
    df = pd.read_csv(data_file_path, skiprows=1)
    
    # Select columns of interest and rename for clarity
    income_col = 'Estimate!!Median household income in the past 12 months (in 2022 inflation-adjusted dollars)'
    cols_of_interest = ['Geography', 'Geographic Area Name', income_col]
    df = df[cols_of_interest]
    
    # Rename the income column for clarity
    rename_mapper = {income_col: 'Median Household Income'}
    df.columns = [rename_mapper.get(col, col) for col in df.columns]
    
    # Add geographic identifiers using utility function
    df = add_geographic_identifiers(df, 'Geography', slice_start=7)
    
    # Add area names using utility function  
    df = add_area_names(df, 'Geographic Area Name')
    
    # Convert numeric columns to proper numeric types
    numeric_columns = ['Median Household Income']
    df = convert_columns_to_numeric(df, numeric_columns)
    
    # Remove original Geography and Geographic Area Name columns
    df = df.drop(columns=['Geography', 'Geographic Area Name'], errors='ignore')
    
    # Select final columns in desired order
    final_columns = [
        "State",
        "County", 
        "Tract",
        "Block Group",
        "State Name",
        "County Name",
        "Median Household Income"
    ]
    
    df = select_final_columns(df, final_columns)
    
    # Clean and validate data
    df = clean_and_validate_data(df, "income data")
    
    return df