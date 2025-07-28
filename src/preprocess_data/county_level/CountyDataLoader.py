from typing_extensions import Literal
import pandas as pd
from .bounding_box_utils import process_raw_county_bounding_box
from .processors.energy_processors import (
    process_raw_wind_data, 
    process_raw_solar_data, 
    process_raw_solar_roof_data
)
from .processors.economic_processors import (
    process_raw_GDP_data,
    process_raw_eia_electric_price_data,
    process_raw_NREL_electric_price_data,
    process_raw_income_data
)
from .processors.demographic_processors import (
    process_raw_education_data,
    process_raw_race_data,
    process_raw_unemployment_data
)
from .processors.geographic_processors import (
    process_raw_number_private_school_data,
    process_raw_rural_urban_data
)
from .processors.political_processors import (
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
        self.rural_urban_data = None

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
        self.rural_urban_data = basic_social_data["rural_urban"]

        # Load parametric data
        self.race_data = self._load_race_data()
        self.election_data = self._load_election_data()
        self.education_data = self._load_education_data()
        self.electric_data = self._load_electric_price_data()

        # Merge all data
        self.merged_data = self._merge_all_data()

        self.merged_data.drop_duplicates(subset=["State", "County Name"], inplace=True, keep=False)
        self.merged_data.dropna(subset=["State", "County Name"], inplace=True)
        self.merged_data.dropna(subset=["STATEFP", "COUNTYFP"], inplace=True)

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
            
        # Merge rural-urban data
        if self.rural_urban_data is not None:
            merged = merged.merge(self.rural_urban_data, on=["State", "County Name"], how="outer")

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
        data_to_load = {
            "private_schools": process_raw_number_private_school_data,
            "income": process_raw_income_data,
            "unemployment": process_raw_unemployment_data,
            "solar_roof": (process_raw_solar_roof_data, {"bounding_box": bounding_box}),
            "gdp": (process_raw_GDP_data, {"population_data_file_path": self.file_paths["population_data"], "bounding_box": bounding_box}),
            "rural_urban": process_raw_rural_urban_data
        }
        dict_to_return = {}
        for key, func in data_to_load.items():
            try:
                if isinstance(func, tuple):
                    # If the function is a tuple, it contains the function and its kwargs
                    func, kwargs = func
                    dict_to_return[key] = func(data_file_path=self.file_paths[key], **kwargs)
                else:
                    dict_to_return[key] = func(data_file_path=self.file_paths[key])

                print(f"✓ Loaded {key} data")
            except Exception as e:
                print(f"Warning: Could not load {key} data: {e}")
                dict_to_return[key] = None

        return dict_to_return

    def _load_race_data(self):
        """Load race data based on the specified race type."""
        try:
            if self.race_type == "decennial":
                processed_data = process_raw_race_data(
                    data_dir_path=self.file_paths["race"], 
                    race_type="decennial"
                )
                print("✓ Loaded decennial race data")
                return processed_data
            elif self.race_type == "ACS":
                processed_data = process_raw_race_data(
                    data_dir_path=self.file_paths["race"], 
                    race_type="ACS"
                )
                print("✓ Loaded ACS race data")
                return processed_data
            else:
                raise ValueError(f"Invalid race type: {self.race_type}")
        except Exception as e:
            print(f"Warning: Could not load race data: {e}")

    def _load_election_data(self):
        """Load election data based on the specified election type."""
        try:
            if self.election_type == "democrat":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="democrat"
                )
                print("✓ Loaded Democrat election data")
                return processed_data
            elif self.election_type == "republican":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="republican"
                )
                print("✓ Loaded Republican election data")
                return processed_data
            elif self.election_type == "other":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="other"
                )
                print("✓ Loaded Other election data")
                return processed_data
            elif self.election_type == "green":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="green"
                )
                print("✓ Loaded Green election data")
                return processed_data
            elif self.election_type == "libertarian":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="libertarian"
                )
                print("✓ Loaded Libertarian election data")
                return processed_data
            elif self.election_type == "all":
                processed_data = process_raw_election_data(
                    data_file_path=self.file_paths["election"], 
                    party="all"
                )
                print("✓ Loaded all election data")
                return processed_data
            else:
                raise ValueError(f"Invalid election type: {self.election_type}")
        except Exception as e:
            print(f"Warning: Could not load election data: {e}")
            return None

    def _load_education_data(self):
        """Load education data based on the specified education type."""
        try:
            if self.education_type == "18-24":
                processed_data = process_raw_education_data(
                    data_file_path=self.file_paths["education"], 
                    age_range="18-24"
                )
                print("✓ Loaded 18-24 education data")
                return processed_data
            elif self.education_type == "25+":
                processed_data = process_raw_education_data(
                    data_file_path=self.file_paths["education"], 
                    age_range="25+"
                )
                print("✓ Loaded 25+ education data")
                return processed_data
            elif self.education_type == "all":
                processed_data = {
                    "18-24": process_raw_education_data(
                        data_file_path=self.file_paths["education"], 
                        age_range="18-24"
                    ),
                    "25+": process_raw_education_data(
                        data_file_path=self.file_paths["education"], 
                        age_range="25+"
                    )
                }
                print("✓ Loaded 18-24 and 25+ education data")
                return processed_data
            else:
                raise ValueError(f"Invalid education type: {self.education_type}")
        except Exception as e:
            print(f"Warning: Could not load education data: {e}")
            return None

    def _load_electric_price_data(self):
        """Load electric data based on the specified electric dataset and customer class."""
        try:
            if self.electric_dataset == "NREL":
                processed_data = process_raw_NREL_electric_price_data(
                    data_file_path=self.file_paths["electric_price_NREL"]
                )
                print("✓ Loaded NREL electric price data")
                return processed_data
            elif self.electric_dataset == "EIA":
                if self.electric_customer_class == "both":
                    processed_data = {
                        "residential": process_raw_eia_electric_price_data(
                            data_file_path=self.file_paths["electric_price_EIA"], 
                            customer_class="residential"
                        ),
                        "commercial": process_raw_eia_electric_price_data(
                            data_file_path=self.file_paths["electric_price_EIA"], 
                            customer_class="commercial"
                        ),
                    }
                    print("✓ Loaded EIA electric price data for both residential and commercial")
                    return processed_data
                elif self.electric_customer_class == "residential":
                    processed_data = process_raw_eia_electric_price_data(
                        data_file_path=self.file_paths["electric_price_EIA"], 
                        customer_class="residential"
                    )
                    print("✓ Loaded EIA electric price data for residential")
                    return processed_data
                elif self.electric_customer_class == "commercial":
                    processed_data = process_raw_eia_electric_price_data(
                        data_file_path=self.file_paths["electric_price_EIA"], 
                        customer_class="commercial"
                    )
                    print("✓ Loaded EIA electric price data for commercial")
                    return processed_data
                else:
                    return process_raw_eia_electric_price_data(
                        data_file_path=self.file_paths["electric_price_EIA"], 
                        customer_class=self.electric_customer_class  # type: ignore
                    )
            else:
                raise ValueError(f"Invalid electric dataset: {self.electric_dataset}")
        except Exception as e:
            print(f"Warning: Could not load electric price data: {e}")
            return None

    def _load_energy_data(self):
        """Load energy-related data (wind, solar)."""
        wind = process_raw_wind_data(
            data_file_path=self.file_paths["wind"], 
            bounding_box=bounding_box
        )
        print("✓ Loaded wind data")
        solar = self._solar_project_data()
        print("✓ Loaded solar data")

        return {"wind": wind, "solar": solar}
    
    def save_to_csv(self, save_type: Literal["individual", "merged", "both"] = "both", 
                    output_dir: str = "data_processed/county_level"):
        """
        Save processed data to CSV files in the data_processed directory.
        
        Args:
            save_type (str): What to save - "individual", "merged", or "both"
                - "individual": Save each dataset separately
                - "merged": Save only the merged dataset
                - "both": Save both individual and merged datasets
            output_dir (str): Output directory relative to project root (default: "data_processed/county_level")
        """
        # Ensure that the data has been loaded already, i.e self.load_data() has been called
        if self.merged_data is None:
            raise ValueError("Data has not been loaded yet. Please call load_data() first.")
        
        import os
        from src.GLOBAL import ROOT_DIR
        
        # Create full output path
        full_output_dir = os.path.join(ROOT_DIR, output_dir)
        
        # Create directory structure
        os.makedirs(full_output_dir, exist_ok=True)
        
        if save_type in ["individual", "both"]:
            self._save_individual_datasets(full_output_dir)
        
        if save_type in ["merged", "both"]:
            self._save_merged_dataset(full_output_dir)
        
        print(f"✓ Data saved to {full_output_dir}")
    
    def _save_individual_datasets(self, output_dir: str):
        """Save individual datasets to separate CSV files."""
        import os
        
        # Create subdirectories for different data types
        subdirs = {
            "demographic": ["race", "education", "unemployment"],
            "economic": ["income", "electric", "gdp"],
            "energy": ["wind", "solar", "solar_roof"],
            "geographic": ["private_schools", "rural_urban"],
            "political": ["election"]
        }
        
        for subdir in subdirs.keys():
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        # Map datasets to their subdirectories and filenames
        dataset_mapping = {
            # Demographic data
            "race_data": ("demographic", "race_processed.csv"),
            "education_data": ("demographic", "education_processed.csv"), 
            "unemployment_data": ("demographic", "unemployment_processed.csv"),
            
            # Economic data
            "income_data": ("economic", "income_processed.csv"),
            "electric_data": ("economic", "electric_price_processed.csv"),
            "gdp_data": ("economic", "gdp_processed.csv"),
            
            # Energy data
            "wind_data": ("energy", "wind_processed.csv"),
            "solar_data": ("energy", "solar_processed.csv"),
            "solar_roof_data": ("energy", "solar_roof_processed.csv"),
            
            # Geographic data
            "private_schools_data": ("geographic", "private_schools_processed.csv"),
            "rural_urban_data": ("geographic", "rural_urban_processed.csv"),
            
            # Political data
            "election_data": ("political", "election_processed.csv")
        }
        
        saved_count = 0
        
        for attr_name, (subdir, filename) in dataset_mapping.items():
            dataset = getattr(self, attr_name, None)
            
            if dataset is not None:
                filepath = os.path.join(output_dir, subdir, filename)
                
                # Handle different data types (DataFrame, dict, etc.)
                if isinstance(dataset, pd.DataFrame):
                    dataset.to_csv(filepath, index=False)
                    saved_count += 1
                    print(f"  ✓ Saved {attr_name} to {subdir}/{filename}")
                    
                elif isinstance(dataset, dict):
                    # Handle dict datasets (like education with multiple age ranges)
                    for key, df in dataset.items():
                        if df is not None and isinstance(df, pd.DataFrame):
                            # Create filename with key suffix
                            base_name = filename.replace('.csv', f'_{key}.csv')
                            dict_filepath = os.path.join(output_dir, subdir, base_name)
                            df.to_csv(dict_filepath, index=False)
                            saved_count += 1
                            print(f"  ✓ Saved {attr_name}[{key}] to {subdir}/{base_name}")
                else:
                    print(f"  ⚠ Skipped {attr_name} (unsupported type: {type(dataset)})")
            else:
                print(f"  - Skipped {attr_name} (not loaded)")
        
        print(f"  📁 Saved {saved_count} individual datasets")
    
    def _save_merged_dataset(self, output_dir: str):
        """Save the merged dataset to a CSV file."""
        import os
        
        # Create merged subdirectory
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        
        if self.merged_data is not None:
            # Create filename with configuration info
            config_suffix = f"{self.race_type}_{self.election_type}_{self.education_type}_{self.solar_type}_{self.electric_dataset}"
            if self.electric_dataset == "EIA":
                config_suffix += f"_{self.electric_customer_class}"
            
            filename = f"county_merged_data_{config_suffix}.csv"
            filepath = os.path.join(merged_dir, filename)
            
            self.merged_data.to_csv(filepath, index=False)
            print(f"  ✓ Saved merged dataset to merged/{filename}")
            print(f"  📊 Dataset shape: {self.merged_data.shape}")
        else:
            print("  ⚠ No merged data to save (run load_data() first)")
    
    def get_save_summary(self):
        """
        Get a summary of what data is available for saving.
        
        Returns:
            dict: Summary of datasets available for saving
        """
        datasets = {
            "race_data": self.race_data,
            "education_data": self.education_data,
            "unemployment_data": self.unemployment_data,
            "income_data": self.income_data,
            "electric_data": self.electric_data,
            "gdp_data": self.gdp_data,
            "wind_data": self.wind_data,
            "solar_data": self.solar_data,
            "solar_roof_data": self.solar_roof_data,
            "private_schools_data": self.private_schools_data,
            "election_data": self.election_data,
            "merged_data": self.merged_data
        }
        
        summary = {
            "available_datasets": {},
            "total_datasets": 0,
            "total_dataframes": 0,
            "configuration": {
                "race_type": self.race_type,
                "election_type": self.election_type,
                "education_type": self.education_type,
                "solar_type": self.solar_type,
                "electric_dataset": self.electric_dataset,
                "electric_customer_class": self.electric_customer_class
            }
        }
        
        for name, data in datasets.items():
            if data is not None:
                summary["total_datasets"] += 1
                
                if isinstance(data, pd.DataFrame):
                    summary["available_datasets"][name] = {
                        "type": "DataFrame",
                        "shape": data.shape
                    }
                    summary["total_dataframes"] += 1
                elif isinstance(data, dict):
                    df_count = sum(1 for v in data.values() if isinstance(v, pd.DataFrame))
                    summary["available_datasets"][name] = {
                        "type": "dict",
                        "sub_dataframes": df_count,
                        "keys": list(data.keys())
                    }
                    summary["total_dataframes"] += df_count
                else:
                    summary["available_datasets"][name] = {
                        "type": str(type(data)),
                        "note": "May not be saveable as CSV"
                    }
        
        return summary
