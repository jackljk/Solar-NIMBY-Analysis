import pandas as pd
from typing import Optional, Union, Literal
import os

from ..utils import convert_to_int, extract_fips_from_geography, finalize_dataset
from src.GLOBAL import FIPS_MAPPING_DF


def process_raw_education_data(data_file_path: str, age_range: Literal["18-24", "25+", 'all'] = 'all'):
    """Process raw education data based on age range.

    Args:
        data_file_path (str): Path to the raw education data CSV file.
        age_range (Literal["18-24", "25+", 'all'], optional): Age range to filter the data. Defaults to 'all'.

    Returns:
        pd.DataFrame: Processed education data.
    """
    raw_education_data = pd.read_csv(data_file_path)
    
    # make the first row the header as first row contains more meaningful column names
    raw_education_data.columns = raw_education_data.iloc[0]
    raw_education_data = raw_education_data[1:]
    # drop the last column due to random nan value
    raw_education_data = raw_education_data.iloc[:, :-1]
    
    if age_range == '18-24':
        return _process_education_data_18_24(raw_education_data)
    elif age_range == '25+':
        return _process_education_data_25_plus(raw_education_data)
    else:
        education_18_24 = _process_education_data_18_24(raw_education_data)
        education_25_plus = _process_education_data_25_plus(raw_education_data)
        # merge both dataframes on "State" and "County Name"
        return pd.merge(
            education_18_24,
            education_25_plus,
            on=["State", "County Name"],
            suffixes=('_18_24', '_25_plus'),
        )
        
        
def _process_education_data_18_24(raw_education_data: pd.DataFrame) -> pd.DataFrame:
    """Process education data for the 18-24 age group.

    Args:
        raw_education_data (pd.DataFrame): Raw education data.

    Returns:
        pd.DataFrame: Processed education data for the 18-24 age group.
    """
    relevant_columns = (
        raw_education_data.columns.str.contains("Estimate")
        & raw_education_data.columns.str.contains("18 to 24 years")
        & raw_education_data.columns.str.contains("Percent")
        & ~raw_education_data.columns.str.contains("Male")
        & ~raw_education_data.columns.str.contains("Female")
    )

    rename_dict = {
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!Less than high school graduate": "18-24 Less than high school graduate",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!High school graduate (includes equivalency)": "18-24 High school graduate",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!Some college or associate's degree": "18-24 Some college or associate's degree",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!Bachelor's degree or higher": "18-24 Bachelor's degree or higher",
    }

    # Select relevant columns and create base dataframe
    filtered_education_data = pd.concat([
        raw_education_data.loc[:, relevant_columns],
        raw_education_data[["Geographic Area Name", "Geography"]]
    ], axis=1)
    
    # extract fips codes 
    filtered_education_data = extract_fips_from_geography(filtered_education_data)

    # Rename columns to more readable names
    filtered_education_data = filtered_education_data.rename(columns=rename_dict)

    
    # Drop unnecessary columns
    processed_education_data = filtered_education_data.drop(
        columns=[
            "Geographic Area Name",
            "Geography",
            "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years",
        ],
        errors='ignore'
    )
    
    # Use finalize_dataset for consistent FIPS mapping and cleanup
    return finalize_dataset(processed_education_data)

def _process_education_data_25_plus(raw_education_data: pd.DataFrame) -> pd.DataFrame:
    """Process education data for the 25+ age group.

    Args:
        raw_education_data (pd.DataFrame): Raw education data.

    Returns:
        pd.DataFrame: Processed education data for the 25+ age group.
    """
    relevant_columns = (
        raw_education_data.columns.str.contains("Estimate")
        & raw_education_data.columns.str.contains("25 years")
        & raw_education_data.columns.str.contains("Percent")
        & ~raw_education_data.columns.str.contains("Male")
        & ~raw_education_data.columns.str.contains("Female")
        & ~raw_education_data.columns.str.contains("MEDIAN")
    )

    rename_dict = {
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Less than 9th grade": "25+ Less than 9th grade",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!9th to 12th grade, no diploma": "25+ 9th to 12th grade, no diploma",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate (includes equivalency)": "25+ High school graduate",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Some college, no degree": "25+ Some college, no degree",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Associate's degree": "25+ Associate's degree",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree": "25+ Bachelor's degree",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Graduate or professional degree": "25+ Graduate or professional degree",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate or higher": "25+ High school graduate or higher",
        "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher": "25+ Bachelor's degree or higher",
    }

    # Select relevant columns and create base dataframe
    filtered_education_data = pd.concat([
        raw_education_data.loc[:, relevant_columns],
        raw_education_data[["Geographic Area Name", "Geography"]]
    ], axis=1)
    
    # extract fips codes
    filtered_education_data = extract_fips_from_geography(filtered_education_data) # type: ignore

    # Rename columns to more readable names
    filtered_education_data = filtered_education_data.rename(columns=rename_dict) # type: ignore

    processed_education_data = filtered_education_data.drop(
        columns=[
            "Geographic Area Name",
            "Geography",
            "Estimate!!Percent!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over",
        ],
        errors='ignore'
    )
    processed_education_data = processed_education_data.merge(
        FIPS_MAPPING_DF,
        on=["FIPS State", "FIPS County"],
        how="inner",
    )
    processed_education_data = processed_education_data.drop(
        columns=["FIPS State", "FIPS County"]
    )
    return processed_education_data


def process_raw_race_data(data_dir_path: str, race_type: Literal['decennial', 'ACS']) -> pd.DataFrame:
    """
    Processes raw race data for the specified survey from the census bureau.
    Args:
        data_file_path (str): Path to directory that contains (race_decennial_raw.csv and race_acs_raw.csv)
        race_type (Literal['decennial', 'ACS']): The type of race data to process
    Returns:
        pd.DataFrame: Processed race data
    """    
    if race_type == 'decennial':
        raw_race_data = pd.read_csv(os.path.join(data_dir_path, "race_decennial_raw.csv"))
        return _process_decennial_race_data(raw_race_data)
    elif race_type == 'ACS':
        raw_race_data = pd.read_csv(os.path.join(data_dir_path, "race_acs_raw.csv"))
        return _process_acs_race_data(raw_race_data)
    else:
        raise ValueError("Invalid race type. Choose either 'decennial' or 'ACS'.")

def _process_decennial_race_data(raw_race_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process decennial census race data.
    
    Args:
        raw_race_data (pd.DataFrame): Raw decennial race data from CSV.
        
    Returns:
        pd.DataFrame: Processed decennial race data with percentages.
    """
    # Set first row as header and clean data
    raw_race_data.columns = raw_race_data.iloc[0]
    race_data = raw_race_data[1:].copy()
    
    # Define columns of interest for decennial data
    columns_of_interest = (
        list(race_data.columns[1:4]) +  # Geographic columns
        [col for col in race_data.columns[:-1] if "Population of one race:!" in col] +
        [" !!Total:!!Not Hispanic or Latino:!!Population of two or more races:!!Population of two races:"]
    )
    
    # Select relevant columns
    race_data_filtered = race_data[columns_of_interest + ["Geography"]].drop(
        columns=["Geographic Area Name"]
    )
    
    # Clean column names
    race_data_filtered.columns = (
        race_data_filtered.columns
        .str.replace("!!Total:", "Total")
        .str.replace("!!Not Hispanic or Latino", "")
        .str.strip()
    )
    
    # Convert to numeric and calculate percentages
    race_data_numeric = race_data_filtered.set_index("Geography")
    race_data_numeric = race_data_numeric.apply(convert_to_int)
    
    # Calculate percentages (divide by total population)
    race_data_percentages = (
        race_data_numeric.div(race_data_numeric["Total"], axis=0)
        .drop(columns=["Total"])
        .reset_index()
    )
    
    # Extract FIPS codes
    race_data_percentages = extract_fips_from_geography(race_data_percentages)
    
    # Rename columns to standardized names
    column_mapping = {
        "Total!!Hispanic or Latino": "Hispanic/Latino",
        "Total:!!Population of one race:!!White alone": "White",
        "Total:!!Population of one race:!!Black or African American alone": "Black/African American",
        "Total:!!Population of one race:!!American Indian and Alaska Native alone": "American Indian/Alaska Native",
        "Total:!!Population of one race:!!Asian alone": "Asian",
        "Total:!!Population of one race:!!Native Hawaiian and Other Pacific Islander alone": "Native Hawaiian/Other Pacific Islander",
        "Total:!!Population of one race:!!Some Other Race alone": "Others",
    }
    
    race_data_cleaned = race_data_percentages.rename(columns=column_mapping)
    
    # Combine 'Others' with two or more races
    two_or_more_races_col = "Total:!!Population of two or more races:!!Population of two races:"
    if two_or_more_races_col in race_data_cleaned.columns:
        race_data_cleaned["Others"] = (
            race_data_cleaned["Others"] + race_data_cleaned[two_or_more_races_col]
        )
        race_data_cleaned = race_data_cleaned.drop(columns=[two_or_more_races_col])
    
    # Merge with FIPS mapping and clean up
    return finalize_dataset(race_data_cleaned)


def _process_acs_race_data(raw_race_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process American Community Survey (ACS) race data.
    
    Args:
        raw_race_data (pd.DataFrame): Raw ACS race data from CSV.
        
    Returns:
        pd.DataFrame: Processed ACS race data with percentages.
    """
    # Set first row as header and clean data
    raw_race_data.columns = raw_race_data.iloc[0]
    race_data = raw_race_data[1:].drop(columns=raw_race_data.columns[-1]).copy() 
    
    # Select estimate columns (exclude last 2 columns which are typically margin of error)
    estimate_columns = [col for col in race_data.columns if "Estimate" in col]
    relevant_estimates = estimate_columns[:len(estimate_columns) - 2]
    
    race_data_estimates = race_data[["Geography"] + relevant_estimates]
    
    # Clean column names
    race_data_estimates.columns = (
        race_data_estimates.columns
        .str.replace("Estimate!!", "")
        .str.replace("Total:!!", "")
    )
    
    # Convert to numeric and calculate percentages
    race_data_numeric = race_data_estimates.set_index("Geography")
    race_data_numeric = race_data_numeric.apply(convert_to_int)
    
    # Calculate percentages (divide by total population)
    race_data_percentages = (
        race_data_numeric.div(race_data_numeric["Total:"], axis=0)
        .drop(columns=["Total:"])
        .reset_index()
    )
    
    # Extract FIPS codes
    race_data_percentages = extract_fips_from_geography(race_data_percentages)
    
    # Create 'Other' category by combining some categories
    race_data_percentages["Other"] = (
        race_data_percentages["Some other race alone"] + 
        race_data_percentages["Two or more races:"]
    )
    
    # Remove original columns that were combined into 'Other'
    race_data_cleaned = race_data_percentages.drop(
        columns=["Geography", "Some other race alone", "Two or more races:"]
    )
    
    # Rename columns to standardized names
    column_mapping = {
        "White alone": "White",
        "Black or African American alone": "Black/African American",
        "American Indian and Alaska Native alone": "American Indian/Alaska Native",
        "Asian alone": "Asian",
        "Native Hawaiian and Other Pacific Islander alone": "Native Hawaiian/Other Pacific Islander",
    }
    
    race_data_cleaned = race_data_cleaned.rename(columns=column_mapping)
    
    # Merge with FIPS mapping and clean up
    return finalize_dataset(race_data_cleaned)



def process_raw_unemployment_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw unemployment data from a CSV file.

    Args:
        data_file_path (str): Path to the raw unemployment data CSV file.

    Returns:
        pd.DataFrame: Processed unemployment data with unemployment rates by county.
    """
    # Load data and set first row as header
    data = pd.read_csv(data_file_path)
    data.columns = data.iloc[0]
    data = data[1:]
    
    # Remove the last column (typically contains NaN values)
    data = data.iloc[:, :-1]

    # Filter columns for population 16 years and over unemployment data
    # Exclude columns with AGE, RACE, or Labor keywords to focus on overall unemployment
    unemployment_columns = [
        col for col in data.columns
        if "Population 16 years and over" in str(col)
        and "AGE" not in str(col)
        and "RACE" not in str(col)
        and "Labor" not in str(col)
    ]
    
    # Select Geography column plus relevant unemployment columns
    filtered_data = data[["Geography"] + unemployment_columns]
    
    # Select the specific columns we need for unemployment analysis
    required_columns = [
        "Geography",
        "Estimate!!Total!!Population 16 years and over",
        "Estimate!!Unemployment rate!!Population 16 years and over"
    ]
    
    # Check if required columns exist and select them
    available_columns = [col for col in required_columns if col in filtered_data.columns]
    unemployment_data = filtered_data[available_columns].copy()
    
    # Rename columns to more readable names
    column_mapping = {
        "Estimate!!Total!!Population 16 years and over": "Total Unemployment",
        "Estimate!!Unemployment rate!!Population 16 years and over": "Unemployment Rate"
    }
    unemployment_data = unemployment_data.rename(columns=column_mapping)
    
    # Extract FIPS codes from Geography column using utility function
    unemployment_data = extract_fips_from_geography(unemployment_data)
    
    # Use finalize_dataset utility to merge with FIPS mapping and clean up
    final_data = finalize_dataset(unemployment_data)
    
    # Convert unemployment data to numeric for calculations
    if "Total Unemployment" in final_data.columns:
        final_data["Total Unemployment"] = pd.to_numeric(
            final_data["Total Unemployment"], errors='coerce'
        )
    if "Unemployment Rate" in final_data.columns:
        final_data["Unemployment Rate"] = pd.to_numeric(
            final_data["Unemployment Rate"], errors='coerce'
        )
    
    # Reorder columns for consistent output
    column_order = ["State", "County Name", "Total Unemployment", "Unemployment Rate"]
    final_columns = [col for col in column_order if col in final_data.columns]
    
    # Use reindex to ensure DataFrame return type
    result = final_data.reindex(columns=final_columns)
    
    return result
