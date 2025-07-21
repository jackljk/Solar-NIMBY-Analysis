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

# Example usage:
"""
# Basic usage with default settings
loader = BlockGroupDataLoader()
merged_data = loader.load_data()

# Custom data directory
loader = BlockGroupDataLoader(data_dir="/path/to/custom/block_group_data")
merged_data = loader.load_data()

# With solar data enabled
loader = BlockGroupDataLoader(solar_type="enabled")
merged_data = loader.load_data()

# Get summary of loaded data
summary = loader.get_summary()
print(summary)

# Access individual datasets
race_data = loader.race_data
education_data = loader.education_data
income_data = loader.income_data
# etc.
"""