from typing import Literal, Optional
import pandas as pd
from .bounding_box_utils import process_raw_block_group_bounding_box
from .processors.demographic_processors import (
    process_raw_race_data,
    process_raw_education_data,
    process_raw_unemployment_data
)
from .processors.economic_processors import (
    process_raw_income_data
)
from .processors.energy_processors import (
    process_raw_solar_data
)
from .processors.political_processors import (
    process_raw_election_data
)


class BlockGroupDataLoader:
    """
    Class to load block group level data for the Solar NIMBY project.
    This class provides a method to load various datasets based on specified parameters.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
    ):
        """
        Initializes the BlockGroupDataLoader with the directory containing data files.

        Args:
            data_dir (str, optional): Directory where block group level data files are stored. 
                                    If None, uses default structure from ROOT_DIR.
        """        
        # Initialize data attributes to None
        self.bounding_box = None
        self.race_data = None
        self.education_data = None
        self.unemployment_data = None
        self.income_data = None
        self.solar_data = None
        self.election_data = None
        self.merged_data = None
        
        # Set up file paths
        self._setup_file_paths(data_dir)
        
        # Load bounding box data
        self._load_bounding_box()
    
    def _setup_file_paths(self, data_dir: Optional[str] = None):
        """
        Set up file paths for block group level data files.
        
        Args:
            data_dir (str, optional): Custom data directory path
        """
        # Use default structure from ROOT_DIR
        from src.GLOBAL import BLOCK_GROUP_LEVEL_DATA_FILES
        
        self.file_paths = BLOCK_GROUP_LEVEL_DATA_FILES
    
    def _load_bounding_box(self):
        """Load bounding box data for block groups."""
        try:
            self.bounding_box = process_raw_block_group_bounding_box(self.file_paths["bounding_boxes"])
        except Exception as e:
            print(f"Warning: Could not load bounding box data: {e}")
            self.bounding_box = None
    
    def load_data(self):
        """
        Loads the block group level data based on the specified parameters.
        Stores each dataset as an attribute and returns the merged dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the merged block group level data.
        """
        # Load demographic data
        self._load_demographic_data()
        
        # Load economic data
        self._load_economic_data()
        
        # Load political data
        self._load_political_data()
        
        # Load energy data
        self._load_energy_data()
        
        # Merge all data
        self.merged_data = self._merge_all_data()
        
        # Clean final merged data
        self._clean_final_data()
        
        return self.merged_data
    
    def _load_demographic_data(self):
        """Load demographic data (race, education, unemployment)."""
        try:
            # Load race data
            self.race_data = process_raw_race_data(self.file_paths["race"])
            print("✓ Loaded race data")
        except Exception as e:
            print(f"Warning: Could not load race data: {e}")
            self.race_data = None
        
        try:
            # Load education data
            self.education_data = process_raw_education_data(self.file_paths["education"])
            print("✓ Loaded education data")
        except Exception as e:
            print(f"Warning: Could not load education data: {e}")
            self.education_data = None
        
        try:
            # Load unemployment data
            self.unemployment_data = process_raw_unemployment_data(self.file_paths["unemployment"])
            print("✓ Loaded unemployment data")
        except Exception as e:
            print(f"Warning: Could not load unemployment data: {e}")
            self.unemployment_data = None
    
    def _load_economic_data(self):
        """Load economic data (income)."""
        try:
            # Load income data
            self.income_data = process_raw_income_data(self.file_paths["income"])
            print("✓ Loaded income data")
        except Exception as e:
            print(f"Warning: Could not load income data: {e}")
            self.income_data = None
    
    def _load_political_data(self):
        """Load political data (election)."""
        try:
            # Load election data
            self.election_data = process_raw_election_data(self.file_paths["election"])
            print("✓ Loaded election data")
        except Exception as e:
            print(f"Warning: Could not load election data: {e}")
            self.election_data = None
    
    def _load_energy_data(self):
        """Load energy data (solar)."""    
        if self.bounding_box is not None:
            try:
                # Load solar data with bounding box
                self.solar_data = process_raw_solar_data(
                    self.file_paths["solar"], 
                    self.file_paths["bounding_boxes"]
                ).rename(columns={
                    "STATEFP": "State",
                    "COUNTYFP": "County",
                    "TRACTCE": "Tract",
                    "BLKGRPCE": "Block Group",
                    "State": "State Name"
                })
                # Print success message
                print("✓ Loaded solar data")
            except Exception as e:
                print(f"Warning: Could not load solar data: {e}")
                self.solar_data = None
        else:
            print("Solar data loading skipped (no bounding box available)")
            self.solar_data = None
    
    def _merge_all_data(self):
        """
        Merges all loaded data into a single DataFrame.

        Returns:
            pd.DataFrame: Merged block group level data.
        """
        # Start with bounding box as base if available, otherwise use first available dataset
        if self.bounding_box is not None:
            merged = self.bounding_box.copy().rename(columns={
                "STATEFP": "State",
                "COUNTYFP": "County",
                "TRACTCE": "Tract",
                "BLKGRPCE": "Block Group",
                "State": "State Name"
            })
            merge_on = ["State", "County", "Tract", "Block Group"]
        else:
            # Find first available dataset to use as base
            base_data = None
            for data in [self.race_data, self.education_data, self.unemployment_data, 
                        self.income_data, self.solar_data, self.election_data]:
                if data is not None:
                    base_data = data
                    break
            
            if base_data is None:
                raise ValueError("No data could be loaded successfully")
            
            merged = base_data.copy()
            merge_on = ["State", "County", "Tract", "Block Group"]
        
        # Merge race data
        if self.race_data is not None:
            if merged is not self.race_data:  # Don't merge with itself
                merged = merged.merge(
                    self.race_data, on=merge_on, how="outer", suffixes=('', '_race')
                )
        
        # Merge education data
        if self.education_data is not None:
            if merged is not self.education_data:
                merged = merged.merge(
                    self.education_data, on=merge_on, how="outer", suffixes=('', '_edu')
                )
        
        # Merge unemployment data
        if self.unemployment_data is not None:
            if merged is not self.unemployment_data:
                merged = merged.merge(
                    self.unemployment_data, on=merge_on, how="outer", suffixes=('', '_unemp')
                )
        
        # Merge income data
        if self.income_data is not None:
            if merged is not self.income_data:
                merged = merged.merge(
                    self.income_data, on=merge_on, how="outer", suffixes=('', '_income')
                )
        
        # Merge election data
        if self.election_data is not None:
            if merged is not self.election_data:
                merged = merged.merge(
                    self.election_data, on=merge_on, how="outer", suffixes=('', '_election')
                )
        
        # Merge solar data
        if self.solar_data is not None:
            if merged is not self.solar_data:
                merged = merged.merge(
                    self.solar_data, on=merge_on, how="outer", suffixes=('', '_solar')
                )
        
        return merged
    
    def _clean_final_data(self):
        """Clean and validate the final merged dataset."""
        if self.merged_data is not None:
            # Remove duplicates based on geographic identifiers
            geo_cols = ["State", "County", "Tract", "Block Group"]
            available_geo_cols = [col for col in geo_cols if col in self.merged_data.columns]
            
            if available_geo_cols:
                # Remove duplicates
                initial_rows = len(self.merged_data)
                self.merged_data = self.merged_data.drop_duplicates(subset=available_geo_cols) # type: ignore
                final_rows = len(self.merged_data)
                
                if initial_rows != final_rows:
                    print(f"Removed {initial_rows - final_rows} duplicate rows")
                
                # Remove rows with missing geographic identifiers
                mask = self.merged_data[available_geo_cols[0]].notna()
                for col in available_geo_cols[1:]:
                    mask = mask & self.merged_data[col].notna()
                
                rows_before = len(self.merged_data)
                self.merged_data = self.merged_data.loc[mask]
                rows_after = len(self.merged_data)
                
                if rows_before != rows_after:
                    print(f"Removed {rows_before - rows_after} rows with missing geographic identifiers")
            
            print(f"Final dataset shape: {self.merged_data.shape}")
    
    def get_summary(self):
        """
        Get a summary of loaded datasets.
        
        Returns:
            dict: Summary information about loaded datasets
        """
        summary = {
            "datasets_loaded": {},
            "total_rows": 0,
            "total_columns": 0,
            "geographic_coverage": {}
        }
        
        # Check which datasets were loaded
        datasets = {
            "race": self.race_data,
            "education": self.education_data, 
            "unemployment": self.unemployment_data,
            "income": self.income_data,
            "election": self.election_data,
            "solar": self.solar_data,
            "bounding_box": self.bounding_box
        }
        
        for name, data in datasets.items():
            if data is not None:
                summary["datasets_loaded"][name] = {
                    "rows": len(data),
                    "columns": len(data.columns)
                }
        
        # Merged data summary
        if self.merged_data is not None:
            summary["total_rows"] = len(self.merged_data)
            summary["total_columns"] = len(self.merged_data.columns)
            
            # Geographic coverage
            geo_cols = ["State", "County", "Tract", "Block Group"]
            for col in geo_cols:
                if col in self.merged_data.columns:
                    summary["geographic_coverage"][col] = self.merged_data[col].nunique()
        
        return summary
    
    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to retrieve.
        
        Returns:
            pd.DataFrame or None: The requested dataset or None if not found.
        """
        datasets = {
            "race": self.race_data,
            "education": self.education_data,
            "unemployment": self.unemployment_data,
            "income": self.income_data,
            "election": self.election_data,
            "solar": self.solar_data,
            "bounding_box": self.bounding_box
        }

        return datasets.get(dataset_name)

    def save_to_csv(self, save_type: Literal["individual", "merged", "both"] = "both", 
                    output_dir: str = "data_processed/block_group_level"):
        """
        Save processed block group data to CSV files in the data_processed directory.
        
        Args:
            save_type (str): What to save - "individual", "merged", or "both"
                - "individual": Save each dataset separately
                - "merged": Save only the merged dataset
                - "both": Save both individual and merged datasets
            output_dir (str): Output directory relative to project root (default: "data_processed/block_group_level")
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
        
        print(f"✓ Block group data saved to {full_output_dir}")
    
    def _save_individual_datasets(self, output_dir: str):
        """Save individual block group datasets to separate CSV files."""
        import os
        
        # Create subdirectories for different data types
        subdirs = {
            "demographic": ["race", "education", "unemployment"],
            "economic": ["income"],
            "energy": ["solar"],
            "political": ["election"],
            "geographic": ["bounding_box"]
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
            
            # Energy data
            "solar_data": ("energy", "solar_processed.csv"),
            
            # Political data
            "election_data": ("political", "election_processed.csv"),
            
            # Geographic data
            "bounding_box": ("geographic", "bounding_box_processed.csv")
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
                    # Handle dict datasets (if any block group data is stored as dict)
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
        
        print(f"  📁 Saved {saved_count} individual block group datasets")
    
    def _save_merged_dataset(self, output_dir: str):
        """Save the merged block group dataset to a CSV file."""
        import os
        
        # Create merged subdirectory
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        
        if self.merged_data is not None:
            # Create filename for block group merged data
            filename = "block_group_merged_data.csv"
            filepath = os.path.join(merged_dir, filename)
            
            self.merged_data.to_csv(filepath, index=False)
            print(f"  ✓ Saved merged block group dataset to merged/{filename}")
            print(f"  📊 Dataset shape: {self.merged_data.shape}")
        else:
            print("  ⚠ No merged data to save (run load_data() first)")
    
    def get_save_summary(self):
        """
        Get a summary of what block group data is available for saving.
        
        Returns:
            dict: Summary of datasets available for saving
        """
        datasets = {
            "race_data": self.race_data,
            "education_data": self.education_data,
            "unemployment_data": self.unemployment_data,
            "income_data": self.income_data,
            "election_data": self.election_data,
            "solar_data": self.solar_data,
            "bounding_box": self.bounding_box,
            "merged_data": self.merged_data
        }
        
        summary = {
            "available_datasets": {},
            "total_datasets": 0,
            "total_dataframes": 0,
            "data_type": "block_group_level"
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

# Example usage:
"""
# Basic usage with default settings
loader = BlockGroupDataLoader()
merged_data = loader.load_data()

# Custom data directory
loader = BlockGroupDataLoader(data_dir="/path/to/custom/block_group_data")
merged_data = loader.load_data()

# Save processed data to CSV files
loader.save_to_csv()  # Save both individual and merged datasets
loader.save_to_csv(save_type="merged")  # Save only merged dataset
loader.save_to_csv(save_type="individual")  # Save only individual datasets
loader.save_to_csv(output_dir="custom_output/block_group")  # Custom output directory

# Get summary of loaded data
summary = loader.get_summary()
print(summary)

# Get summary of data available for saving
save_summary = loader.get_save_summary()
print(save_summary)

# Access individual datasets
race_data = loader.race_data
education_data = loader.education_data
income_data = loader.income_data
solar_data = loader.solar_data
election_data = loader.election_data
bounding_box = loader.bounding_box

# Get specific dataset by name
race_data = loader.get_dataset("race")
solar_data = loader.get_dataset("solar")
"""