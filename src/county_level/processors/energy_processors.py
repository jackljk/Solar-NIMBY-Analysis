import geopandas as gpd
import pandas as pd
from src.county_level.utils import merge_data


# Constants for column naming
TOTAL_PREFIX = "total"
AVG_PREFIX = "avg"
COUNT_PREFIX = "count"
INTENSITY_SUFFIX = "Intensity (MW/ 1000 sq mile)"
PROJECT_INTENSITY_SUFFIX = "Project Intensity (Projects/ 1000 sq mile)"


def _aggregate_energy_data(data, groupby_columns, aggregation_column):
    """
    Aggregates energy data by specified columns and computes sum, mean, and count statistics.

    Parameters:
        data (pd.DataFrame): The input data to aggregate.
        groupby_columns (list): The columns to group by.
        aggregation_column (str): The column to aggregate.

    Returns:
        pd.DataFrame: The aggregated data with total, average, and count columns.
    """
    aggregation_dict = {
        f"{TOTAL_PREFIX}_{aggregation_column}": (aggregation_column, "sum"),
        f"{AVG_PREFIX}_{aggregation_column}": (aggregation_column, "mean"),
        f"{COUNT_PREFIX}_{aggregation_column}": (aggregation_column, "count"),
    }

    return data.groupby(groupby_columns).agg(**aggregation_dict).reset_index()


def _calculate_project_metrics(data, energy_source):
    """
    Calculate project metrics such as area in square miles and square kilometers.

    Parameters:
        data (pd.DataFrame): The input data containing geometry.

    Returns:
        pd.DataFrame: The data with calculated metrics.
    """
    area_column = "area mi2"
    source_capitalized = energy_source.capitalize()

    # Calculate capacity intensity (MW per 1000 square miles)
    capacity_intensity_col = f"{source_capitalized} Capacity {INTENSITY_SUFFIX}"
    data[capacity_intensity_col] = (
        data[f"{TOTAL_PREFIX}_{energy_source}_mw"] / data[area_column] * 1000
    )

    # Calculate project intensity (Projects per 1000 square miles)
    project_intensity_col = f"{source_capitalized} {PROJECT_INTENSITY_SUFFIX}"
    data[project_intensity_col] = (
        data[f"{COUNT_PREFIX}_{energy_source}_mw"] / data[area_column] * 1000
    )

    # Calculate average capacity intensity (Average MW per 1000 square miles)
    avg_intensity_col = f"{source_capitalized} Avg Capacity {INTENSITY_SUFFIX}"
    data[avg_intensity_col] = (
        data[f"{AVG_PREFIX}_{energy_source}_mw"] / data[area_column] * 1000
    )

    # Clean up intermediate columns
    columns_to_drop = [
        "GEOID",
        f"{TOTAL_PREFIX}_{energy_source}_mw",
        f"{AVG_PREFIX}_{energy_source}_mw",
        f"{COUNT_PREFIX}_{energy_source}_mw",
        "area mi2",
        "area km2",
        "FIPS State",
        "FIPS County",
        "STATEFP",
        "COUNTYFP",
    ]

    # Only drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    return (
        data.drop(columns=existing_columns_to_drop)
        if existing_columns_to_drop
        else data
    )


def process_raw_wind_data(data_file_path, bounding_box=None):
    """
    Process raw wind energy data from a geospatial file.

    Parameters:
        data_file_path (str): Path to the raw wind data file.
        bounding_box (pd.DataFrame): Bounding box data to filter the results geographically.

    Returns:
        pd.DataFrame: Processed wind data with intensity metrics.

    Raises:
        ValueError: If required parameters are not provided.
    """
    # _validate_processing_inputs(data_file_path, bounding_box)

    # Load and select required columns from wind data
    required_columns = ["county", "statename", "wind_mw"]
    raw_wind_data = gpd.read_file(data_file_path)[required_columns]

    # Define column mapping for standardization
    column_mapping = {
        "statename": "State",
        "county": "County Name",
        "STATEFP": "FIPS State",
        "COUNTYFP": "FIPS County",
    }

    # Aggregate wind data by state and county
    aggregated_data = _aggregate_energy_data(
        data=raw_wind_data,
        groupby_columns=["statename", "county"],
        aggregation_column="wind_mw",
    ).rename(columns=column_mapping)

    # merge with bounding box for geographic filtering
    merged_data = merge_data(
        data=aggregated_data,
        on_columns=["State", "County Name"],
        bounding_box=bounding_box
    )

    # Calculate project metrics
    processed_data = _calculate_project_metrics(data=merged_data, energy_source="wind")
    # Replace NaN values with 0 for numeric columns
    numeric_columns = processed_data.select_dtypes(include=["number"]).columns
    processed_data[numeric_columns] = processed_data[numeric_columns].fillna(0)

    # replace NaN values in string columns with 0 as well
    string_columns = processed_data.select_dtypes(include=["object"]).columns
    processed_data[string_columns] = processed_data[string_columns].fillna("0")

    return processed_data


def process_raw_solar_data(data_file_path, bounding_box=None, size="all"):
    """Process raw solar energy data from a geospatial file.

    Args:
        data_file_path (str): Path to the raw solar data file.
        bounding_box (pd.DataFrame, optional): Bounding box data to filter the results geographically. Defaults to None.

    Returns:
        pd.DataFrame: Processed solar data with intensity metrics.
    """
    # load and select required columns from solar data
    required_columns = ["county", "statename", "solar_mw"]
    solar_data = gpd.read_file(data_file_path)[required_columns]

    # ensure that solar_mw is of type float
    solar_data["solar_mw"] = solar_data["solar_mw"].astype(float)

    if size == "all":
        # No filtering applied, use all data
        pass
    elif size == "small":
        solar_data = solar_data[solar_data["solar_mw"] < 5]
    elif size == "medium":
        solar_data = solar_data[
            (solar_data["solar_mw"] >= 5) & (solar_data["solar_mw"] < 25)
        ]
    elif size == "large":
        solar_data = solar_data[solar_data["solar_mw"] >= 25]
    else:
        raise ValueError(f"Invalid size type: {size}")

    # Define column mapping for standardization
    column_mapping = {
        "statename": "State",
        "county": "County Name",
        "STATEFP": "FIPS State",
        "COUNTYFP": "FIPS County",
    }

    # Aggregate solar data by state and county
    aggregated_data = _aggregate_energy_data(
        data=solar_data,
        groupby_columns=["statename", "county"],
        aggregation_column="solar_mw",
    ).rename(columns=column_mapping)

    # Merge with bounding box for geographic filtering
    merged_data = merge_data(
        data=aggregated_data,
        on_columns=["State", "County Name"],
        bounding_box=bounding_box
    )


    # Calculate project metrics
    processed_data = _calculate_project_metrics(data=merged_data, energy_source="solar")

    # Replace NaN values with 0 for numeric columns
    numeric_columns = processed_data.select_dtypes(include=["number"]).columns
    processed_data[numeric_columns] = processed_data[numeric_columns].fillna(0)

    # replace NaN values in string columns with 0 as well
    string_columns = processed_data.select_dtypes(include=["object"]).columns
    processed_data[string_columns] = processed_data[string_columns].fillna("0")
    return processed_data


def process_raw_solar_roof_data(data_file_path, bounding_box=None):
    """
    Process raw solar roof data from a geospatial file.

    Parameters:
        data_file_path (str): Path to the raw solar roof data file.
        bounding_box (pd.DataFrame): Bounding box data to filter the results geographically.

    Returns:
        pd.DataFrame: Processed solar roof data with intensity metrics.
    """
    # Load and select required columns from solar roof data
    required_columns = [
        "region_name",
        "state_name",
        "existing_installs_count",
        "kw_total",
        "kw_median",
    ]
    solar_roof_data = pd.read_csv(data_file_path)[required_columns]
    
    # fix typing for kw_total and kw_median
    solar_roof_data["kw_total"] = solar_roof_data["kw_total"].astype(float)
    solar_roof_data["kw_median"] = solar_roof_data["kw_median"].astype(float)

    column_mapping = {
        "region_name": "County Name",
        "state_name": "State",
        "existing_installs_count": "Number of Existing Installs",
        "kw_total": "Total Installed Capacity (kW)",
        "kw_median": "Median Installed Capacity (kW)",
    }
    # Get the sum of existing installs count, total installed capacity, and median installed capacity
    solar_roof_data = (
        solar_roof_data.groupby(["region_name", "state_name"])
        .sum()
        .reset_index()
        .rename(columns=column_mapping)
    )

    # unique case for solar roof dataset renaming
    solar_roof_data['County Name'] = solar_roof_data['County Name'].str.replace(' County', '').str.replace(' Parish', '').str.replace(".", "")
    
    # merge with bounding box for geographic filtering
    merged_data = merge_data(
        data=solar_roof_data,
        on_columns=["State", "County Name"],
        bounding_box=bounding_box
    )
    
    # Calculate project metrics manually due to different structure
    processed_data = merged_data.copy()
    processed_data['Total Installed Capacity (kW/ 1000 sq mile)'] = processed_data['Total Installed Capacity (kW)'] / processed_data['area mi2'] * 1000
    processed_data['Median Installed Capacity (kW/ 1000 sq mile)'] = processed_data['Median Installed Capacity (kW)'] / processed_data['area mi2'] * 1000
    processed_data['Number of Existing Installs (Projects/ 1000 sq mile)'] = (
        processed_data['Number of Existing Installs'] / processed_data['area mi2'] * 1000
    )
    
    # round the calculated columns to 2 decimal places
    processed_data['Total Installed Capacity (kW/ 1000 sq mile)'] = processed_data['Total Installed Capacity (kW/ 1000 sq mile)'].round(2)
    processed_data['Median Installed Capacity (kW/ 1000 sq mile)'] = processed_data['Median Installed Capacity (kW/ 1000 sq mile)'].round(2)
    processed_data['Number of Existing Installs (Projects/ 1000 sq mile)'] = processed_data['Number of Existing Installs (Projects/ 1000 sq mile)'].round(2)
    
    # Clean up intermediate columns
    columns_to_drop = [
        "GEOID",
        "area mi2",
        "area km2",
        "FIPS State",
        "FIPS County",
        "STATEFP",
        "COUNTYFP",
    ]
    # Only drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in processed_data.columns]

    processed_data = processed_data.drop(columns=existing_columns_to_drop)

    # Replace NaN values with 0 for numeric columns
    numeric_columns = processed_data.select_dtypes(include=["number"]).columns
    processed_data[numeric_columns] = processed_data[numeric_columns].fillna(0)
    
    # replace NaN values in string columns with 0 as well
    string_columns = processed_data.select_dtypes(include=["object"]).columns
    processed_data[string_columns] = processed_data[string_columns].fillna("0")
    return processed_data