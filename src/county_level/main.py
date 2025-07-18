from typing_extensions import Literal
import pandas as pd
from src.county_level.bounding_box_utils import process_raw_county_bounding_box
from src.county_level.processors.energy_processors import (
    process_raw_wind_data, 
    process_raw_solar_data, 
    process_raw_solar_roof_data
)
from src.county_level.processors.economic_processors import (
    process_raw_GDP_data,
    process_raw_eia_electric_price_data,
    process_raw_NREL_electric_price_data,
    process_raw_income_data
)
from src.county_level.processors.demographic_processors import (
    process_raw_education_data,
    process_raw_race_data,
    process_raw_unemployment_data
)
from src.county_level.processors.geographic_processors import (
    process_raw_number_private_school_data
)
from src.county_level.processors.political_processors import (
    process_raw_election_data
)

# Load bounding box data
bounding_box = process_raw_county_bounding_box("data/bounding_boxes/county_bounding_box_raw")


class CountyDataLoader:
    """
    Class to load county-level data for the Solar NIMBY project.
    This class provides a method to load various datasets based on specified parameters.
    """

    def __init__(
        self,
        race_type: Literal["decennial", "ACE"] = "decennial",
        election_type: Literal['all', 'democrat', 'republican', 'green', 'libertarian', 'other'] = 'all',
        education_type: Literal["18-24", "25-34", "all"] = "all",
        solar_type: Literal["all", "all_only", "small_only", "medium_only", "large_only"] = "all",
        electric_customer_class: Literal["both", "residential", "commercial"] = "both",
        electric_dataset: Literal["NREL", "EIA"] = "NREL",
    ):
        """
        Initializes the CountyDataLoader class.
        """
        self.race_type = race_type
        self.election_type = election_type
        self.education_type = education_type
        self.solar_type = solar_type
        self.electric_customer_class = electric_customer_class
        self.electric_dataset = electric_dataset

        # Initialize data attributes to None
        self.wind_data = None
        self.gdp_data = None
        self.solar_data = None
        self.private_schools_data = None
        self.income_data = None
        self.unemployment_data = None
        self.solar_roof_data = None
        self.race_data = None
        self.election_data = None
        self.education_data = None
        self.electric_data = None
        self.merged_data = None

        # get all the file paths for the datasets
        from src.GLOBAL import COUNTY_LEVEL_DATA_FILES
        self.file_paths = COUNTY_LEVEL_DATA_FILES

    def load_data(self):
        """
        Loads the county-level data based on the specified parameters.
        Stores each dataset as an attribute and returns the merged dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the merged county-level data.
        """
        # Load energy data
        energy_data = self._load_energy_data()
        self.wind_data = energy_data["wind"]
        self.solar_data = energy_data["solar"]

        # Load basic social data
        basic_social_data = self._load_basic_social_data()
        self.private_schools_data = basic_social_data["private_schools"]
        self.income_data = basic_social_data["income"]
        self.unemployment_data = basic_social_data["unemployment"]
        self.solar_roof_data = basic_social_data["solar_roof"]
        self.gdp_data = basic_social_data["gdp"]

        # Load parametric data
        self.race_data = self._load_race_data()
        self.election_data = self._load_election_data()
        self.education_data = self._load_education_data()
        self.electric_data = self._load_electric_price_data()

        # Merge all data
        self.merged_data = self._merge_all_data()

        return self.merged_data

    def _merge_all_data(self):
        """
        Merges all loaded data into a single DataFrame.

        Returns:
            pd.DataFrame: Merged county-level data.
        """
        # Start with bounding box data as the base
        merged = bounding_box.copy()
        
        # Merge energy data
        if self.wind_data is not None:
            merged = merged.merge(
                self.wind_data, on=["State", "County Name"], how="outer"
            )
        if self.gdp_data is not None:
            merged = merged.merge(
                self.gdp_data, on=["State", "County Name"], how="outer"
            )
        
        # Merge solar data (handle dict case for different sizes)
        if self.solar_data is not None:
            if isinstance(self.solar_data, dict):
                for size, solar_df in self.solar_data.items():
                    if solar_df is not None:
                        merged = merged.merge(
                            solar_df, on=["State", "County Name"], how="outer", suffixes=('', f'_{size}')
                        )
            else:
                merged = merged.merge(
                    self.solar_data, on=["State", "County Name"], how="outer"
                )

        # Merge basic social data
        if self.private_schools_data is not None:
            merged = merged.merge(
                self.private_schools_data, on=["State", "County Name"], how="outer"
            )
        if self.income_data is not None:
            merged = merged.merge(
                self.income_data, on=["State", "County Name"], how="outer"
            )
        if self.unemployment_data is not None:
            merged = merged.merge(
                self.unemployment_data, on=["State", "County Name"], how="outer"
            )
        if self.solar_roof_data is not None:
            merged = merged.merge(
                self.solar_roof_data, on=["State", "County Name"], how="outer"
            )
        if self.race_data is not None:
            merged = merged.merge(self.race_data, on=["State", "County Name"], how="outer")

        # Merge election data (handle dict case)
        if self.election_data is not None:
            if isinstance(self.election_data, dict):
                for key, election_df in self.election_data.items():
                    if election_df is not None:
                        merged = merged.merge(
                            election_df, on=["State", "County Name"], how="outer"
                        )
            else:
                merged = merged.merge(
                    self.election_data, on=["State", "County Name"], how="outer"
                )

        # Merge education data (handle dict case)
        if self.education_data is not None:
            if isinstance(self.education_data, dict):
                for key, education_df in self.education_data.items():
                    if education_df is not None:
                        merged = merged.merge(
                            education_df, on=["State", "County Name"], how="outer"
                        )
            else:
                merged = merged.merge(
                    self.education_data, on=["State", "County Name"], how="outer"
                )

        # Merge electric data (handle EIA dict case)
        if self.electric_data is not None:
            if self.electric_dataset == "NREL":
                if isinstance(self.electric_data, pd.DataFrame):
                    merged = merged.merge(
                        self.electric_data, on=["State", "County Name"], how="outer"
                    )
            elif self.electric_dataset == "EIA":
                if self.electric_customer_class == "both" and isinstance(
                    self.electric_data, dict
                ):
                    for key, electric_df in self.electric_data.items():
                        if electric_df is not None and isinstance(electric_df, pd.DataFrame):
                            merged = merged.merge(
                                electric_df, on=["State", "County Name"], how="outer"
                            )
                else:
                    if isinstance(self.electric_data, pd.DataFrame):
                        merged = merged.merge(
                            self.electric_data, on=["State", "County Name"], how="outer"
                        )

        return merged

    # split the load_data method into smaller methods for better readability and maintainability as part of the class
    def _solar_project_data(self):
        """
        Loads solar data based on the specified solar type.

        Returns:
            dict: A dictionary containing solar data for different sizes.
        """
        if self.solar_type == "all":
            solar_all = {
                "all": process_raw_solar_data(
                    data_file_path=self.file_paths["solar"], 
                    bounding_box=bounding_box
                ),
                "small": process_raw_solar_data(
                    data_file_path=self.file_paths["solar"],
                    bounding_box=bounding_box,
                    size="small"
                ),
                "medium": process_raw_solar_data(
                    data_file_path=self.file_paths["solar"],
                    bounding_box=bounding_box,
                    size="medium"
                ),
                "large": process_raw_solar_data(
                    data_file_path=self.file_paths["solar"],
                    bounding_box=bounding_box,
                    size="large"
                )
            }
            return solar_all
        elif self.solar_type in ["all_only", "small_only", "medium_only", "large_only"]:
            size = self.solar_type.replace("_only", "")
            return process_raw_solar_data(
                data_file_path=self.file_paths["solar"],
                bounding_box=bounding_box,
                size=size
            )
        else:
            raise ValueError(f"Invalid solar type: {self.solar_type}")

    def _load_basic_social_data(self):
        """Load basic social factor data that doesn't require parameter choices."""
        return {
            "private_schools": process_raw_number_private_school_data(
                data_file_path=self.file_paths["private_schools"]
            ),
            "income": process_raw_income_data(
                data_file_path=self.file_paths["income"]
            ),
            "unemployment": process_raw_unemployment_data(
                data_file_path=self.file_paths["unemployment"]
            ),
            "solar_roof": process_raw_solar_roof_data(
                data_file_path=self.file_paths["solar_roof"], 
                bounding_box=bounding_box
            ),
            "gdp": process_raw_GDP_data(
                data_file_path=self.file_paths["gdp"], 
                population_data_file_path=self.file_paths["population_data"],
                bounding_box=bounding_box
            )
        }

    def _load_race_data(self):
        """Load race data based on the specified race type."""
        if self.race_type == "decennial":
            return process_raw_race_data(
                data_dir_path=self.file_paths["race"], 
                race_type="decennial"
            )
        elif self.race_type == "ACS":
            return process_raw_race_data(
                data_dir_path=self.file_paths["race"], 
                race_type="ACS"
            )
        else:
            raise ValueError(f"Invalid race type: {self.race_type}")

    def _load_election_data(self):
        """Load election data based on the specified election type."""
        if self.election_type == "democrat":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="democrat"
            )
        elif self.election_type == "republican":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="republican"
            )
        elif self.election_type == "other":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="other"
            )
        elif self.election_type == "green":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="green"
            )
        elif self.election_type == "libertarian":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="libertarian"
            )
        elif self.election_type == "all":
            return process_raw_election_data(
                data_file_path=self.file_paths["election"], 
                party="all"
            )
        else:
            raise ValueError(f"Invalid election type: {self.election_type}")

    def _load_education_data(self):
        """Load education data based on the specified education type."""
        if self.education_type == "18-24":
            return process_raw_education_data(
                data_file_path=self.file_paths["education"], 
                age_range="18-24"
            )
        elif self.education_type == "25+":
            return process_raw_education_data(
                data_file_path=self.file_paths["education"], 
                age_range="25+"
            )
        elif self.education_type == "all":
            return {
                "18-24": process_raw_education_data(
                    data_file_path=self.file_paths["education"], 
                    age_range="18-24"
                ),
                "25+": process_raw_education_data(
                    data_file_path=self.file_paths["education"], 
                    age_range="25+"
                ),
            }
        else:
            raise ValueError(f"Invalid education type: {self.education_type}")

    def _load_electric_price_data(self):
        """Load electric data based on the specified electric dataset and customer class."""
        if self.electric_dataset == "NREL":
            return process_raw_NREL_electric_price_data(
                data_file_path=self.file_paths["electric_price_NREL"]
            )
        elif self.electric_dataset == "EIA":
            if self.electric_customer_class == "both":
                return {
                    "residential": process_raw_eia_electric_price_data(
                        data_file_path=self.file_paths["electric_price_EIA"], 
                        customer_class="residential"
                    ),
                    "commercial": process_raw_eia_electric_price_data(
                        data_file_path=self.file_paths["electric_price_EIA"], 
                        customer_class="commercial"
                    ),
                }
            else:
                return process_raw_eia_electric_price_data(
                    data_file_path=self.file_paths["electric_price_EIA"], 
                    customer_class=self.electric_customer_class  # type: ignore
                )
        else:
            raise ValueError(f"Invalid electric dataset: {self.electric_dataset}")

    def _load_energy_data(self):
        """Load energy-related data (wind, GDP, solar)."""
        wind = process_raw_wind_data(
            data_file_path=self.file_paths["wind"], 
            bounding_box=bounding_box
        )
        solar = self._solar_project_data()

        return {"wind": wind, "solar": solar}
