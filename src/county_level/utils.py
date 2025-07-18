import pandas as pd
from src.GLOBAL import FIPS_MAPPING_DF


def merge_data(data: pd.DataFrame, on_columns: list[str], how='left', **dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Merges data with another DataFrame on specified columns.

    Parameters:
        data (pd.DataFrame): The input data to merge.
        on_columns (list): The columns to merge on.
        how (str): The type of merge to perform (default is 'left').
        **dataframes: Additional DataFrames to merge with the input data.
    Returns:
        pd.DataFrame: The merged data.
    """
    merged_data = data
    for df in dataframes.values():
        merged_data = merged_data.merge(df, on=on_columns, how=how)
    return merged_data


def finalize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the dataset by merging with FIPS mapping and cleaning up. (Reused in multiple dataset cleaning functions)
    
    Args:
        data (pd.DataFrame): Processed race data with FIPS codes.
        
    Returns:
        pd.DataFrame: Final cleaned race data.
    """
    # Merge with FIPS mapping
    final_data = data.merge(
        FIPS_MAPPING_DF,
        on=["FIPS State", "FIPS County"],
        how="inner",
    )
    
    # Remove unnecessary columns
    columns_to_drop = [
        "FIPS State", "FIPS County", "Geography", "Geography Name",
    ]
    
    # Drop columns if they exist in the DataFrame
    final_data = final_data.drop(columns=[col for col in columns_to_drop if col in final_data.columns])
    
    return final_data

def convert_to_int(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to integer, handling non-numeric values.
    
    Args:
        series (pd.Series): The series to convert.

    Returns:
        pd.Series: The converted series with numeric values.
    """
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)


def extract_fips_from_geography(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract FIPS State and County codes from Geography column.
    
    Args:
        data (pd.DataFrame): DataFrame with Geography column.
        
    Returns:
        pd.DataFrame: DataFrame with FIPS State and FIPS County columns added.
    """
    data = data.copy()
    
    # Handle case-insensitive column name
    geography_col = 'Geography' if 'Geography' in data.columns else 'geography'
    
    # Extract FIPS codes from Geography column (format: "...US{STATE}{COUNTY}")
    data["FIPS State"] = data[geography_col].str.extract(r'US(\d{2})\d{3}')[0]
    data["FIPS County"] = data[geography_col].str.extract(r'US\d{2}(\d{3})')[0]
    
    return data