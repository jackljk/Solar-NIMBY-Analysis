import pandas as pd
from ..utils import (
    parse_geo_codes, to_numeric,
    add_geographic_identifiers, add_area_names, convert_columns_to_numeric,
    clean_and_validate_data, select_final_columns
)


def process_raw_race_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw race data for the specified survey from the census bureau.
    
    NOTE: There is only one survey Decennial Census, so this function is designed to handle that only.
    
    Args:
        data_file_path (str): Path to race data files
    Returns:
        pd.DataFrame: Processed race data
    """
    # Read from first row (skip header row)
    df = pd.read_csv(data_file_path, skiprows=1)
    
    # Define columns of interest
    col_of_interest = (
        ['Geography'] + 
        list(df.columns[1:4]) + 
        [col for col in df.columns[:-1] if '!!Total:!!Population of one race:!' in col] + 
        [" !!Total:!!Population of two or more races:"]
    )
    
    # Select only columns of interest
    df = df[col_of_interest]
    
    # Define rename mapping for cleaner column names
    rename_mapper = {
        " !!Total:": "Total Population",
        " !!Total:!!Population of one race:": "Total One Race",
        " !!Total:!!Population of one race:!!White alone": "White Only",
        " !!Total:!!Population of one race:!!Black or African American alone": "African American Only",
        " !!Total:!!Population of one race:!!American Indian and Alaska Native alone": "American Indian and Alaska Native Only",
        " !!Total:!!Population of one race:!!Asian alone": "Asian Only",
        " !!Total:!!Population of one race:!!Native Hawaiian and Other Pacific Islander alone": "Native Hawaiian and Other Pacific Islander Only",
        " !!Total:!!Population of one race:!!Some Other Race alone": "Others Only",
        " !!Total:!!Population of two or more races:": "Total Mixed Race"
    }
    
    # Rename columns
    df.columns = [rename_mapper.get(col, col) for col in df.columns]
    
    # Parse geographic codes using utility function
    df['Geography'] = df['Geography'].str[7:]  # Remove 'Block Group: ' prefix
    geo_codes = df['Geography'].apply(parse_geo_codes)
    df[['State', 'County', 'Tract', 'Block Group']] = pd.DataFrame(geo_codes.tolist(), index=df.index)
    
    # Extract county and state names if Geographic Area Name column exists
    df = add_area_names(df, 'Geographic Area Name')
    
    # Convert numeric columns to proper numeric types
    numeric_columns = [
        "Total Population",
        "Total One Race", 
        "White Only",
        "African American Only",
        "American Indian and Alaska Native Only",
        "Asian Only",
        "Native Hawaiian and Other Pacific Islander Only",
        "Others Only",
        "Total Mixed Race"
    ]
    
    df = convert_columns_to_numeric(df, numeric_columns)
    
    # Select final columns in desired order
    final_columns = [
        "State",
        "County", 
        "Tract",
        "Block Group",
        "State Name",
        "County Name",
        "Total Population",
        "Total One Race",
        "White Only",
        "African American Only", 
        "American Indian and Alaska Native Only",
        "Asian Only",
        "Native Hawaiian and Other Pacific Islander Only",
        "Others Only",
        "Total Mixed Race"
    ]
    
    df = select_final_columns(df, final_columns)
    
    df = clean_and_validate_data(df, "race data")
    
    return df



def process_raw_education_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw education data for block group level.
    
    Args:
        data_file_path (str): Path to education data file
    Returns:
        pd.DataFrame: Processed education data
    """
    # Read the raw education data
    raw_education_data = pd.read_csv(data_file_path, skiprows=1)
    
    # Get all columns that have "Estimate" in the name
    estimate_cols = raw_education_data.columns[raw_education_data.columns.str.contains('Estimate')]
    cols = ['Geography'] + list(estimate_cols)
    
    # Select only the columns we need
    df_edu = raw_education_data[cols]
    
    # Define education level categories
    less_than_9th_grade = [
        'No schooling completed', 'Nursery school', 'Kindergarten', 
        '1st grade', '2nd grade', '3rd grade', '4th grade', 
        '5th grade', '6th grade', '7th grade', '8th grade'
    ]
    grade_9th_to_12th_no_diploma = [
        '9th grade', '10th grade', '11th grade', '12th grade, no diploma'
    ]
    high_school_graduate = [
        'Regular high school diploma', 'GED or alternative credential'
    ]
    some_college_no_degree = [
        'Some college, less than 1 year', 'Some college, 1 or more years, no degree'
    ]
    associate_degree = ["Associate's degree"]
    bachelor_degree = ["Bachelor's degree"]
    graduate_degree = [
        "Master's degree", 'Professional school degree', 'Doctorate degree'
    ]
    
    # Create aggregate categories
    highschool_or_higher = (high_school_graduate + some_college_no_degree + 
                           associate_degree + bachelor_degree + graduate_degree)
    bach_or_higher = bachelor_degree + graduate_degree
    
    # Find matching columns for each education category
    less_than_9th_grade_cols = [col for col in cols if any(grade in col for grade in less_than_9th_grade)]
    grade_9th_to_12th_no_diploma_cols = [col for col in cols if any(grade in col for grade in grade_9th_to_12th_no_diploma)]
    high_school_graduate_cols = [col for col in cols if any(grade in col for grade in high_school_graduate)]
    some_college_no_degree_cols = [col for col in cols if any(grade in col for grade in some_college_no_degree)]
    associate_degree_cols = [col for col in cols if any(grade in col for grade in associate_degree)]
    bachelor_degree_cols = [col for col in cols if any(grade in col for grade in bachelor_degree)]
    graduate_degree_cols = [col for col in cols if any(grade in col for grade in graduate_degree)]
    highschool_or_higher_cols = [col for col in cols if any(grade in col for grade in highschool_or_higher)]
    bach_or_higher_cols = [col for col in cols if any(grade in col for grade in bach_or_higher)]
    
    # Find the total population column
    total_col = 'Estimate!!Total:'
    
    # Create the processed education DataFrame
    edu_df = pd.DataFrame()
    edu_df['Geography'] = df_edu['Geography']
    
    # Parse geographic codes using utility function
    # Clean geography column by removing common prefixes
    edu_df = add_geographic_identifiers(edu_df, 'Geography')
    
    # Calculate education percentages
    total_population = to_numeric(df_edu[total_col])
    
    # Helper function to calculate percentage for each education category
    def calculate_percentage(column_list):
        if column_list:
            # Convert to numeric and sum across rows manually
            subset = df_edu[column_list]
            # Apply to_numeric to each column then sum manually
            numeric_subset = subset.apply(lambda col: to_numeric(col))
            
            if len(column_list) == 1:
                row_sums = numeric_subset.iloc[:, 0]
            else:
                # Manual sum across columns to avoid axis issues
                row_sums = numeric_subset.iloc[:, 0]
                for i in range(1, len(column_list)):
                    row_sums = row_sums + numeric_subset.iloc[:, i]
                    
            return (row_sums / total_population * 100)
        else:
            return pd.Series([0] * len(df_edu), index=df_edu.index)
    
    # Calculate percentages for each education category
    edu_df['less_than_9th_grade'] = calculate_percentage(less_than_9th_grade_cols)
    edu_df['grade_9th_to_12th_no_diploma'] = calculate_percentage(grade_9th_to_12th_no_diploma_cols)
    edu_df['high_school_graduate'] = calculate_percentage(high_school_graduate_cols)
    edu_df['some_college_no_degree'] = calculate_percentage(some_college_no_degree_cols)
    edu_df['associate_degree'] = calculate_percentage(associate_degree_cols)
    edu_df['bachelor_degree'] = calculate_percentage(bachelor_degree_cols)
    edu_df['graduate_degree'] = calculate_percentage(graduate_degree_cols)
    edu_df['highschool_or_higher'] = calculate_percentage(highschool_or_higher_cols)
    edu_df['bach_or_higher'] = calculate_percentage(bach_or_higher_cols)
    
    # Remove Geography column as we have parsed the codes
    edu_df = edu_df.drop(columns=['Geography'])
    
    # Remove rows with missing geographic identifiers
    mask = (edu_df['State'].notna() & edu_df['County'].notna() & 
           edu_df['Tract'].notna() & edu_df['Block Group'].notna())
    result_df = edu_df.loc[mask]
    
    # Ensure we return a DataFrame
    if isinstance(result_df, pd.DataFrame):
        return result_df
    else:
        # If somehow we don't get a DataFrame, create one
        return pd.DataFrame(result_df).T if hasattr(result_df, '__iter__') else pd.DataFrame()


def process_raw_unemployment_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw unemployment data for block group level.
    
    Args:
        data_file_path (str): Path to unemployment data file
    Returns:
        pd.DataFrame: Processed unemployment data
    """
    # Read the raw unemployment data
    df = pd.read_csv(data_file_path, skiprows=1)
    
    # Select columns of interest - Geography and all Estimate columns
    cols = ["Geography", "Geographic Area Name"] + [col for col in df.columns if "Estimate" in col]
    df = df[cols]
    
    # Define rename mapping for cleaner column names
    rename_mapper = {
        "Estimate!!Total:": "Total Population",
        "Estimate!!Total:!!In labor force:": "In Labor Force",
        "Estimate!!Total:!!In labor force:!!Civilian labor force:": "Civilian Labor Force",
        "Estimate!!Total:!!In labor force:!!Civilian labor force:!!Employed": "Employed",
        "Estimate!!Total:!!In labor force:!!Civilian labor force:!!Unemployed": "Unemployed",
        "Estimate!!Total:!!In labor force:!!Armed Forces": "Armed Forces",
        "Estimate!!Total:!!Not in labor force": "Not in Labor Force"
    }
    
    # Rename columns using the mapping
    df.columns = [rename_mapper.get(col, col) for col in df.columns]
    
    # Parse geographic codes using utility function
    # Clean geography column by removing common prefixes
    df = add_geographic_identifiers(df, 'Geography')
    
    # Extract county and state names using utility functions
    df = add_area_names(df, 'Geographic Area Name')
    
    # Convert numeric columns to proper numeric types
    numeric_columns = [
        "Total Population",
        "In Labor Force",
        "Civilian Labor Force", 
        "Employed",
        "Unemployed",
        "Armed Forces",
        "Not in Labor Force"
    ]
    
    df = convert_columns_to_numeric(df, numeric_columns)
    
    # Calculate unemployment rate
    if 'Unemployed' in df.columns and 'In Labor Force' in df.columns:
        # Avoid division by zero
        labor_force = df['In Labor Force']
        unemployed = df['Unemployed']
        df['Unemployment Rate'] = unemployed / labor_force.replace(0, pd.NA)
    else:
        df['Unemployment Rate'] = pd.NA
    
    
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
        "Total Population",
        "In Labor Force",
        "Civilian Labor Force",
        "Employed", 
        "Unemployed",
        "Armed Forces",
        "Not in Labor Force",
        "Unemployment Rate",
    ]
    
    df = select_final_columns(df, final_columns)
    
    df = clean_and_validate_data(df, "unemployment data")
    
    return df

