import pandas as pd
from typing import Optional, Union, Literal
from src.county_level.utils import merge_data, extract_fips_from_geography, finalize_dataset

from src.GLOBAL import FIPS_MAPPING_DF, EIA_FIPS_MAPPING_DF


def _extract_fips(GeoFIPS: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Extracts County and State FIPS codes from GeoFIPS.

    Args:
        GeoFIPS (pd.DataFrame | pd.Series): DataFrame or Series containing GeoFIPS codes.

    Returns:
        pd.DataFrame: DataFrame with extracted 'County FIPS' and 'State FIPS'.
    """
    GeoFIPS = GeoFIPS.str.strip().str.replace('"', "")
    return pd.DataFrame({"County FIPS": GeoFIPS.str[2:], "State FIPS": GeoFIPS.str[:2]})


def process_raw_GDP_data(
    data_file_path: str,
    population_data_file_path: str,
    bounding_box: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Processes raw GDP data from a CSV file, filtering by bounding box and merging with population data if provided.

    Args:
        data_file_path (str): Path to the raw GDP data CSV file.
        bounding_box (Optional[pd.DataFrame]): DataFrame containing bounding box coordinates for filtering.
        population_data (Optional[pd.DataFrame]): DataFrame containing population data for merging.

    Returns:
        pd.DataFrame: Processed GDP data.
    """
    assert bounding_box is not None, "Bounding box must be provided for filtering."
    assert (
        population_data_file_path is not None
    ), "Population data must be provided for merging."

    raw_GDP_data = pd.read_csv(data_file_path, dtype={"GEOFIPS": str})
    population_data = pd.read_csv(
        population_data_file_path, dtype={"STATE": str, "COUNTY": str}
    )
    # Fix the 'Description' column in raw GDP data
    raw_GDP_data["Description"] = raw_GDP_data["Description"].str.strip()
    raw_GDP_data = raw_GDP_data[
        raw_GDP_data["Description"] == "Real GDP (thousands of chained 2017 dollars)"
    ]
    # Extract FIPS codes from GeoFIPS
    raw_GDP_data[["COUNTYFP", "STATEFP"]] = _extract_fips(raw_GDP_data["GeoFIPS"])
    # rename columns for population data
    population_data = population_data.rename(
        columns={
            "COUNTY": "COUNTYFP",
            "STATE": "STATEFP",
        }
    )

    # merge with bounding box and population data
    raw_GDP_data = merge_data(
        data=raw_GDP_data,  # type: ignore
        on_columns=["STATEFP", "COUNTYFP"],
        how="inner",
        bounding_box=bounding_box,
        population_data=population_data,
    )

    # Filter and rename columns
    GDP_data = raw_GDP_data[
        [
            "STATEFP",
            "COUNTYFP",
            "State",
            "County Name",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "POPESTIMATE2022",
        ]
    ]

    rename_dict = {
        "2017": "GDP_2017",
        "2018": "GDP_2018",
        "2019": "GDP_2019",
        "2020": "GDP_2020",
        "2021": "GDP_2021",
        "2022": "GDP_2022",
        "POPESTIMATE2022": "Population Estimate 2022",
    }
    GDP_data = GDP_data.rename(columns=rename_dict) # type: ignore
    GDP_data = GDP_data.drop(columns=["STATEFP", "COUNTYFP"])

    # normalize GDP data by population estimate
    for col in ["GDP_2017", "GDP_2018", "GDP_2019", "GDP_2020", "GDP_2021", "GDP_2022"]:
        GDP_data[col] = (
            GDP_data[col].astype(float) / GDP_data["Population Estimate 2022"]
        )
        # round to 2 decimal places
        GDP_data[col] = GDP_data[col].round(2)

    return GDP_data.drop(
        columns=["Population Estimate 2022"]
    ).reset_index(drop=True)


def process_raw_eia_electric_price_data(
    data_file_path: str, customer_class: Literal["residential", "commercial", "both"]
) -> Union[pd.DataFrame, dict[Literal["commercial", "residential"], pd.DataFrame]]:
    """
    Processes raw electric price data from a CSV file for a specific customer class.

    Args:
        data_file_path (str): Path to the raw electric price data CSV file.
        customer_class (Literal["residential", "commercial", "both"]): The customer class to filter by.

    Returns:
        pd.DataFrame: Processed electric price data.
    """
    relevant_columns = [
        "utility_id_eia",
        "customer_class",
        "customers",
        "sales_mwh",
        "sales_revenue",
    ]
    raw_electric_price_data = pd.read_csv(
        data_file_path, dtype={"utility_id_eia": str}
    )[relevant_columns]

    FIPS_MAPPING_DF_FULL = FIPS_MAPPING_DF.copy()
    FIPS_MAPPING_DF_FULL["FIPS Full"] = (
        FIPS_MAPPING_DF_FULL["FIPS State"].astype(int).astype(str)
        + FIPS_MAPPING_DF_FULL["FIPS County"]
    ).drop(columns=["FIPS State", "FIPS County"])

    raw_electric_price_data = raw_electric_price_data.merge(
        EIA_FIPS_MAPPING_DF, on="utility_id_eia", how="left"
    ).drop(columns=["utility_id_eia"])

    raw_electric_price_data = raw_electric_price_data.merge(
        FIPS_MAPPING_DF_FULL, left_on="county_id_fips", right_on="FIPS Full", how="left"
    )

    # drop duplicate rows looking at 'county_id_fips'
    raw_electric_price_data = raw_electric_price_data.drop_duplicates(
        subset=["county_id_fips", "customer_class"]
    )

    commercial_data = raw_electric_price_data[
        raw_electric_price_data["customer_class"] == "commercial"
    ]
    residential_data = raw_electric_price_data[
        raw_electric_price_data["customer_class"] == "residential"
    ]

    commercial_data_summed = (
        commercial_data.groupby("county_id_fips")
        .sum()
        .reset_index()
        .drop(columns=["customer_class"])
    )
    residential_data_summed = (
        residential_data.groupby("county_id_fips")
        .sum()
        .reset_index()
        .drop(columns=["customer_class"])
    )

    # rename columns
    commercial_data_rename_dict = {
        "sales_revenue": "Commercial Sales Revenue",
        "sales_mwh": "Commercial Sales MWH",
        "customers": "No. Commercial Customers",
    }
    residential_data_rename_dict = {
        "sales_revenue": "Residential Sales Revenue",
        "sales_mwh": "Residential Sales MWH",
        "customers": "No. Residential Customers",
    }

    commercial_data_summed = commercial_data_summed.rename(
        columns=commercial_data_rename_dict
    ).drop(columns=["FIPS Full", "county_id_fips", "FIPS State", "FIPS County"])
    residential_data_summed = residential_data_summed.rename(
        columns=residential_data_rename_dict
    ).drop(columns=["FIPS Full", "county_id_fips", "FIPS State", "FIPS County"])
    if customer_class == "commercial":
        return commercial_data_summed
    elif customer_class == "residential":
        return residential_data_summed
    elif customer_class == "both":
        return {
            "commercial": commercial_data_summed,
            "residential": residential_data_summed,
        }


def process_raw_NREL_electric_price_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw NREL electric price data from a CSV file.

    Args:
        data_file_path (str): Path to the raw NREL electric price data CSV file.

    Returns:
        pd.DataFrame: Processed NREL electric price data.
    """
    raw_electric_price_data = pd.read_csv(data_file_path)

    average_electric_price_data = (
        raw_electric_price_data[
            ["State", "County Name", "comm_rate", "ind_rate", "res_rate"]
        ]
        .groupby(["State", "County Name"])
        .mean()
        .reset_index()
    )
    rename_dict = {
        "comm_rate": "Electric Commercial Rate",
        "ind_rate": "Electric Industrial Rate",
        "res_rate": "Electric Residential Rate"
    }
    average_electric_price_data = average_electric_price_data.rename(columns=rename_dict)
    return average_electric_price_data

def process_raw_income_data(data_file_path: str) -> pd.DataFrame:
    """
    Processes raw income data from a CSV file.

    Args:
        data_file_path (str): Path to the raw income data CSV file.

    Returns:
        pd.DataFrame: Processed income data with median household income by county.
    """
    # Load and clean raw data
    raw_data = pd.read_csv(data_file_path)

    # Set first row as header and remove it from data
    raw_data.columns = raw_data.iloc[0]
    raw_data = raw_data[1:].drop(columns=raw_data.columns[-1]) 

    # Filter to relevant income columns
    estimate_columns = [
        col for col in raw_data.columns
        if all(keyword in str(col) for keyword in ["Estimate", "Households", "Median"])
    ]
    relevant_columns = ["Geography"] + estimate_columns
    income_data = raw_data.reindex(columns=relevant_columns).rename(
        columns={"Estimate!!Households!!Median income (dollars)": "Median Income"}
    )
    
    
    # Extract FIPS codes from Geography column
    income_data = extract_fips_from_geography(income_data)
    
    # Merge with FIPS mapping and finalize
    income_data = finalize_dataset(income_data).drop_duplicates()

    return income_data

