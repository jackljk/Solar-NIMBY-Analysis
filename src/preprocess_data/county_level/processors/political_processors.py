import pandas as pd
from typing import Union, Literal, Dict
from src.GLOBAL import FIPS_MAPPING_DF

# Constants for political parties
MAJOR_PARTIES = ["DEMOCRAT", "REPUBLICAN"]
MINOR_PARTIES = ["GREEN", "LIBERTARIAN"]


def process_raw_election_data(
    data_file_path: str, 
    party: Literal['all', 'democrat', 'republican', 'green', 'libertarian', 'other'] = 'all'
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Processes raw election data from a CSV file, filtering by party if specified.

    Args:
        data_file_path (str): Path to the raw election data CSV file.
        party (Literal): The political party to filter by or 'all' for all parties.

    Returns:
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]: 
            - If party is 'all': Dictionary with party names as keys and DataFrames as values
            - Otherwise: Single DataFrame for the specified party
    """
    # Load and clean raw data
    raw_data = pd.read_csv(data_file_path, dtype={"county_fips": str})

    # Extract FIPS codes and select relevant columns
    processed_data = _extract_fips_and_filter_columns(raw_data)
    
    # Aggregate votes by county and party
    aggregated_data = _aggregate_votes_by_county_and_party(processed_data)
    
    # Merge with FIPS mapping to get state and county names
    merged_data = _merge_with_fips_mapping(aggregated_data)
    
    # Calculate vote percentages
    vote_percentages = _calculate_vote_percentages(merged_data)
    
    # Create party-specific datasets
    party_datasets = _create_party_datasets(vote_percentages)
    
    # Return based on party parameter
    return _filter_by_party(party_datasets, party)


def _extract_fips_and_filter_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Extract FIPS codes and select relevant columns."""
    data = data.copy()
    
    # Clean county_fips by removing last 2 characters
    data["county_fips"] = data["county_fips"].str[:-2]
    
    # Extract State and County FIPS codes
    data["FIPS State"] = data["county_fips"].str[:-3].str.strip()
    data["FIPS County"] = data["county_fips"].str[-3:].str.strip()
    
    # Select relevant columns
    relevant_columns = [
        "FIPS County", "FIPS State", "candidate", "mode", 
        "party", "candidatevotes", "totalvotes"
    ]
    
    # Select relevant columns - using list indexing to ensure DataFrame return type
    result = data.reindex(columns=relevant_columns).copy()
    
    return result


def _aggregate_votes_by_county_and_party(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate votes by county and party."""
    return (
        data.groupby(["FIPS County", "FIPS State", "party"])
        .agg({
            "candidatevotes": "sum",
            "totalvotes": "sum"
        })
        .reset_index()
    )


def _merge_with_fips_mapping(data: pd.DataFrame) -> pd.DataFrame:
    """Merge with FIPS mapping to get state and county names."""
    # Ensure FIPS State is string type for consistent merging
    fips_mapping = FIPS_MAPPING_DF.copy()
    fips_mapping["FIPS State"] = fips_mapping["FIPS State"].astype(int).astype(str)
    
    merged_data = data.merge(
        fips_mapping,
        on=["FIPS State", "FIPS County"],
        how="inner"
    )
    
    # Remove FIPS columns as they're no longer needed
    return merged_data.drop(columns=["FIPS State", "FIPS County"])


def _calculate_vote_percentages(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate vote percentages and clean up vote count columns."""
    data["percentage_vote"] = data["candidatevotes"] / data["totalvotes"]
    return data.drop(columns=["candidatevotes", "totalvotes"])


def _create_party_datasets(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create separate datasets for each party with vote percentages."""
    party_datasets = {}
    
    # Define party mappings
    party_filters = {
        "democrat": data["party"] == "DEMOCRAT",
        "republican": data["party"] == "REPUBLICAN", 
        "green": data["party"] == "GREEN",
        "libertarian": data["party"] == "LIBERTARIAN",
        "other": ~data["party"].isin(["DEMOCRAT", "REPUBLICAN", "GREEN", "LIBERTARIAN"])
    }
    
    # Create dataset for each party
    for party_name, party_filter in party_filters.items():
        party_data = data[party_filter][["State", "County Name", "percentage_vote"]].copy()
        # Rename the percentage_vote column
        new_column_name = f"{party_name}_percentage_vote"
        party_data.columns = ["State", "County Name", new_column_name]
        party_datasets[party_name] = party_data
    
    return party_datasets


def _filter_by_party(
    party_datasets: Dict[str, pd.DataFrame], 
    party: str
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Filter results based on requested party."""
    if party == "all":
        return party_datasets
    
    party_lower = party.lower()
    if party_lower in party_datasets:
        return party_datasets[party_lower]
    else:
        raise ValueError(
            f"Invalid party '{party}'. Valid options: 'all', 'democrat', 'republican', "
            f"'green', 'libertarian', 'other'"
        )

