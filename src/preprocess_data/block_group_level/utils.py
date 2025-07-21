"""
Utility functions for data preprocessing and geographic code parsing.
"""
from typing import Union, Tuple, Optional, Any, List
import pandas as pd
import re


def to_numeric(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Convert string numbers with commas to numeric format.
    
    Args:
        data: A pandas Series or DataFrame containing string numbers with commas
        
    Returns:
        The same data structure with numeric values
        
    Raises:
        ValueError: If data cannot be converted to numeric format
    """
    def clean_numeric_string(value: Any) -> Any:
        """Helper function to clean individual values."""
        try:
            if pd.isna(value):
                return value
            if isinstance(value, str):
                # Remove commas, dollar signs, and other common currency/number formatting
                cleaned = re.sub(r'[,$%]', '', value.strip())
                # Handle parentheses for negative numbers (accounting format)
                if cleaned.startswith('(') and cleaned.endswith(')'):
                    cleaned = '-' + cleaned[1:-1]
                return cleaned
            return value
        except Exception:
            return value
    
    try:
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            for col in result.columns:
                if result[col].dtype == 'object':
                    result[col] = pd.to_numeric(result[col].apply(clean_numeric_string), errors='coerce')
            return result
        else:
            cleaned_data = data.apply(clean_numeric_string)
            converted = pd.to_numeric(cleaned_data, errors='coerce')
            # Ensure we return a Series, not a scalar
            if isinstance(converted, pd.Series):
                return converted
            else:
                # If conversion resulted in a scalar, create a Series
                return pd.Series([converted], index=data.index if hasattr(data, 'index') else [0])
    except Exception as e:
        raise ValueError(f"Error converting data to numeric: {e}")


def parse_geo_codes(geo_id: str) -> Tuple[str, str, str, str]:
    """
    Parse geographic codes from a GEOID string.
    
    Args:
        geo_id: A geographic identifier string (e.g., 'US01001950100')
        
    Returns:
        Tuple of (state_code, county_code, tract_code, block_group_code)
        
    Raises:
        ValueError: If the geo_id format is invalid
    """
    if not isinstance(geo_id, str):
        raise ValueError(f"geo_id must be a string, got {type(geo_id)}")
    
    if not geo_id.startswith('US'):
        raise ValueError(f"geo_id must start with 'US', got: {geo_id}")
    
    # Remove 'US' prefix
    codes = geo_id[2:]
    
    # Validate minimum length (state + county + tract + block group = 2 + 3 + 6 + 1 = 12)
    if len(codes) < 12:
        raise ValueError(f"geo_id too short, expected at least 12 digits after 'US', got {len(codes)}: {geo_id}")
    
    state_code = codes[:2]
    county_code = codes[2:5]
    tract_code = codes[5:11]  # Census tracts are 6 digits
    block_group_code = codes[11]  # Block group is 1 digit
    
    return state_code, county_code, tract_code, block_group_code


def extract_county_name(geo_area_name: str, delimiter: str = ',') -> Optional[str]:
    """
    Extract county name from a geographic area name string.
    
    Args:
        geo_area_name: Geographic area name in format like "Block Group 1, Census Tract 123, County Name, State"
        
    Returns:
        County name or None if extraction fails
    """
    if not isinstance(geo_area_name, str):
        return None

    parts = [part.strip() for part in geo_area_name.split(delimiter)]

    # Expected format: Block Group X, Census Tract Y, County Name, State
    if len(parts) >= 3:
        county_part = parts[2]
        # Remove common county suffixes for consistency
        county_name = re.sub(r'\s+(County|Parish|Borough|City and Borough|Municipality)$', '', county_part, flags=re.IGNORECASE)
        return county_name.strip()
    
    return None


def extract_state_name(geo_area_name: str, delimiter: str = ',') -> Optional[str]:
    """
    Extract state name from a geographic area name string.
    
    Args:
        geo_area_name: Geographic area name in format like "Block Group 1, Census Tract 123, County Name, State"
        
    Returns:
        State name or None if extraction fails
    """
    if not isinstance(geo_area_name, str):
        return None
    
    parts = [part.strip() for part in geo_area_name.split(delimiter)]
    
    # State should be the last part
    if len(parts) >= 4:
        return parts[3].strip()
    
    return None


def add_geographic_identifiers(df: pd.DataFrame, geography_col: str = 'Geography', 
                              slice_start: int = 7) -> pd.DataFrame:
    """
    Add geographic identifiers (State, County, Tract, Block Group) to DataFrame.
    
    Args:
        df: DataFrame containing geography column
        geography_col: Name of the geography column
        slice_start: Position to slice from to remove prefix 
    
    Returns:
        DataFrame with added geographic identifier columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if slice_start is not None:
        # Use slicing method (for specific data formats)
        clean_geography = result_df[geography_col].str[slice_start:]
    else:
        raise ValueError("Slice start must be provided for slicing geography strings.")
    
    geo_codes = clean_geography.apply(parse_geo_codes)
    result_df[['State', 'County', 'Tract', 'Block Group']] = pd.DataFrame(geo_codes.tolist(), index=result_df.index)
    return result_df


def add_area_names(df: pd.DataFrame, area_name_col: str = 'Geographic Area Name') -> pd.DataFrame:
    """
    Add county and state names using geographic area name column.
    
    Args:
        df: DataFrame containing geographic area name column
        area_name_col: Name of the geographic area name column
    
    Returns:
        DataFrame with added County Name and State Name columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if area_name_col in result_df.columns:
        result_df['County Name'] = result_df[area_name_col].apply(extract_county_name, delimiter=';')
        result_df['State Name'] = result_df[area_name_col].apply(extract_state_name, delimiter=';')
    else:
        result_df['County Name'] = None
        result_df['State Name'] = None
    return result_df


def convert_columns_to_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric types.
    
    Args:
        df: DataFrame to process
        numeric_columns: List of column names to convert
    
    Returns:
        DataFrame with converted numeric columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = to_numeric(result_df[col])
    return result_df


def clean_and_validate_data(df: pd.DataFrame, data_type: str = "data") -> pd.DataFrame:
    """
    Common data cleaning and validation steps for block group data.
    
    Args:
        df: DataFrame to clean
        data_type: Type of data (for logging purposes)
    
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check for duplicates and remove if any
    duplicates = result_df.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: Found {duplicates} duplicate rows in {data_type}, removing...")
        result_df = result_df.drop_duplicates()
    
    # Remove rows with missing geographic identifiers
    geo_columns = ['State', 'County', 'Tract', 'Block Group']
    available_geo_cols = [col for col in geo_columns if col in result_df.columns]
    
    if available_geo_cols:
        # Create mask for rows with all geographic identifiers present
        mask = True
        for col in available_geo_cols:
            mask = mask & result_df[col].notna()
        filtered = result_df.loc[mask]
        return filtered if isinstance(filtered, pd.DataFrame) else result_df
    
    return result_df


def select_final_columns(df: pd.DataFrame, desired_columns: List[str]) -> pd.DataFrame:
    """
    Select only available columns from desired list.
    
    Args:
        df: DataFrame to filter
        desired_columns: List of desired column names
    
    Returns:
        DataFrame with only available desired columns
    """
    available_columns = [col for col in desired_columns if col in df.columns]
    result = df[available_columns]
    # Ensure we return a DataFrame, not a Series
    if isinstance(result, pd.DataFrame):
        return result
    else:
        return pd.DataFrame(result).T


# Backward compatibility aliases
def to_int(df):
    """Deprecated: Use to_numeric instead."""
    import warnings
    warnings.warn("to_int is deprecated, use to_numeric instead", DeprecationWarning, stacklevel=2)
    return to_numeric(df)


def get_codes(geo):
    """Deprecated: Use parse_geo_codes instead."""
    import warnings
    warnings.warn("get_codes is deprecated, use parse_geo_codes instead", DeprecationWarning, stacklevel=2)
    return parse_geo_codes(geo)


def get_county_name(geo_area_name):
    """Deprecated: Use extract_county_name instead."""
    import warnings
    warnings.warn("get_county_name is deprecated, use extract_county_name instead", DeprecationWarning, stacklevel=2)
    return extract_county_name(geo_area_name)


def get_state_name(geo_area_name):
    """Deprecated: Use extract_state_name instead."""
    import warnings
    warnings.warn("get_state_name is deprecated, use extract_state_name instead", DeprecationWarning, stacklevel=2)
    return extract_state_name(geo_area_name)
