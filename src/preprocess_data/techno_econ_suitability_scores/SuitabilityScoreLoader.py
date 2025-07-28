from typing import Literal, Optional
import os
import geopandas as gpd
import pandas as pd

from src.preprocess_data.techno_econ_suitability_scores.utils import (
    process_tif_files,
    handle_Connecticut_county_mapping,
    handle_code_matching_error,
    SUITABILITY_SCORES_FILES
)


class SuitabilityScoreLoader:
    def __init__(
        self,
        geographic_level: Literal["all", "county", "block_group", "project"] = "all",
        file_paths=None,
    ):
        self.geographic_level = geographic_level
        if file_paths is None:
            self.file_paths = SUITABILITY_SCORES_FILES.values()
        else:
            self.file_paths = file_paths

        self.suitability_scores = {}

    def load_suitability_scores(self):
        """
        Load suitability scores based on the geographic level.
        """
        if self.geographic_level == "all":
            print("Processing suitability scores for all geographic levels...")
            print("Processing county suitability scores...")
            self.suitability_scores["county"] = self._load_county_suitability_scores()
            print("✓ Loaded county suitability data")
            print("Processing block group suitability scores...")
            self.suitability_scores["block_group"] = (
                self._load_block_group_suitability_scores()
            )
            print("✓ Loaded block group suitability data")
            print("Processing project suitability scores...")
            self.suitability_scores["project"] = self._load_project_suitability_scores()
            print("✓ Loaded project suitability data")
        elif self.geographic_level == "project":
            print("Processing project suitability scores...")
            self.suitability_scores["project"] = self._load_project_suitability_scores()
            print("✓ Loaded project suitability data")
        elif self.geographic_level == "county":
            print("Processing county suitability scores...")
            self.suitability_scores["county"] = self._load_county_suitability_scores()
            print("✓ Loaded county suitability data")
        elif self.geographic_level == "block_group":
            print("Processing block group suitability scores...")
            self.suitability_scores["block_group"] = (
                self._load_block_group_suitability_scores()
            )
            print("✓ Loaded block group suitability data")
        else:
            raise ValueError("Invalid geographic level specified.")

        return self.suitability_scores

    def _load_county_suitability_scores(self):
        """
        Load county suitability scores.
        """
        # load the raw county bounding box to get the 'geometry' column
        from src.preprocess_data.county_level.bounding_box_utils import (
            process_raw_county_bounding_box,
        )
        from src.GLOBAL import COUNTY_LEVEL_DATA_FILES

        county_bounding_box_dir = COUNTY_LEVEL_DATA_FILES["bounding_boxes"]

        bounding_box = process_raw_county_bounding_box(county_bounding_box_dir)
        raw_bounding_box = gpd.read_file(
            os.path.join(county_bounding_box_dir, "cb_2018_us_county_500k.shp"),
            dtype={"GEOID": str},
        )[["GEOID", "geometry"]]

        merged_bounding_box = raw_bounding_box.merge(
            bounding_box, on="GEOID", how="left"
        ).to_crs(epsg=4326)

        # Process the TIF files for county level suitability scores
        county_suitability_scores = process_tif_files(
            tif_filepaths=self.file_paths,
            bounding_box=merged_bounding_box,
            block_group=False,
        )
        
        # dropna
        county_suitability_scores = county_suitability_scores.dropna(
            subset=["State", "County Name"]
        )
        
        return county_suitability_scores

    def _load_block_group_suitability_scores(self):
        """
        Load block group suitability scores.
        """
        # load the raw block group bounding box to get the 'geometry' column
        from src.preprocess_data.block_group_level.bounding_box_utils import (
            process_raw_block_group_bounding_box,
        )
        from src.GLOBAL import BLOCK_GROUP_LEVEL_DATA_FILES

        block_group_bounding_box_dir = BLOCK_GROUP_LEVEL_DATA_FILES["bounding_boxes"]

        bounding_box = process_raw_block_group_bounding_box(
            block_group_bounding_box_dir
        )
        raw_bounding_box = gpd.read_file(
            os.path.join(block_group_bounding_box_dir, "cb_2023_us_bg_500k.shp"),
            dtype={"GEOID": str},
        )[["GEOID", "geometry"]]

        merged_bounding_box = raw_bounding_box.merge(
            bounding_box, on="GEOID", how="left"
        ).to_crs(epsg=4326)

        # Process the TIF files for block group level suitability scores
        block_group_suitability_scores = process_tif_files(
            tif_filepaths=self.file_paths,
            bounding_box=merged_bounding_box,
            block_group=True,
        )
        # Handle Connecticut county mapping
        block_group_suitability_scores = handle_Connecticut_county_mapping(
            block_group_suitability_scores
        )
        
        # Handle code matching errors
        block_group_suitability_scores = handle_code_matching_error(
            block_group_suitability_scores
        )
        # dropna
        block_group_suitability_scores = block_group_suitability_scores.dropna(
            subset=["State", "County Name"]
        )
        return block_group_suitability_scores

    def _load_project_suitability_scores(self):
        from src.preprocess_data.block_group_level.bounding_box_utils import (
            process_raw_block_group_bounding_box,
        )
        
        from src.preprocess_data.techno_econ_suitability_scores.utils import (
            process_tif_files_with_average,
            get_GEOID_from_point,
        )
        
        from src.GLOBAL import BLOCK_GROUP_LEVEL_DATA_FILES

        block_group_bounding_box_dir = BLOCK_GROUP_LEVEL_DATA_FILES["bounding_boxes"]

        bounding_box = gpd.read_file(
            os.path.join(block_group_bounding_box_dir, "cb_2023_us_bg_500k.shp"), dtype={'GEOID': str, 'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str, 'BLKGRPCE': str})
        
        raw_solar_data = pd.read_csv(
            BLOCK_GROUP_LEVEL_DATA_FILES["solar"],
        )
        # rename wkt column to geometry
        raw_solar_data.rename(columns={"WKT": "geometry"}, inplace=True)

        suitability_data = process_tif_files_with_average(self.file_paths, raw_solar_data, nodata_value=255)
        
        suitability_data[['GEOID', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE']] = suitability_data['geometry'].apply(lambda x: get_GEOID_from_point(x, bounding_box)).apply(pd.Series)

        return suitability_data.dropna()

    def save_to_csv(self, output_dir: str = "data_processed/suitability_scores",
                    geographic_levels: Optional[list] = None):
        """
        Save processed suitability scores to CSV files in the data_processed directory.
        
        Args:
            output_dir (str): Output directory relative to project root (default: "data_processed/suitability_scores")
            geographic_levels (list, optional): List of geographic levels to save. If None, saves all loaded levels.
        """
        # Ensure that the data has been loaded already
        if not self.suitability_scores:
            raise ValueError("Suitability scores have not been loaded yet. Please call load_suitability_scores() first.")
        
        import os
        from src.GLOBAL import ROOT_DIR
        
        # Create full output path
        full_output_dir = os.path.join(ROOT_DIR, output_dir)
        
        # Create directory structure
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Determine which geographic levels to save
        if geographic_levels is None:
            levels_to_save = list(self.suitability_scores.keys())
        else:
            levels_to_save = [level for level in geographic_levels if level in self.suitability_scores]
            if not levels_to_save:
                print("Warning: None of the specified geographic levels are available in loaded data.")
                return
        
        self._save_suitability_datasets(full_output_dir, levels_to_save)
        
        print(f"✓ Suitability scores saved to {full_output_dir}")
    
    def _save_suitability_datasets(self, output_dir: str, levels_to_save: list):
        """Save suitability datasets directly to the output directory without subdirectories."""
        import os
        
        saved_count = 0
        
        for level in levels_to_save:
            level_data = self.suitability_scores[level]
            
            if level_data is not None and isinstance(level_data, pd.DataFrame):
                # Create filename based on geographic level
                filename = f"{level}_suitability_scores.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save the data
                level_data.to_csv(filepath, index=False)
                saved_count += 1
                print(f"  ✓ Saved {level} suitability scores to {filename}")
                print(f"    📊 Dataset shape: {level_data.shape}")
            else:
                print(f"  ⚠ Skipped {level} (no valid DataFrame data)")
        
        print(f"  📁 Saved {saved_count} suitability datasets")
    def get_save_summary(self):
        """
        Get a summary of what suitability score data is available for saving.
        
        Returns:
            dict: Summary of datasets available for saving
        """
        summary = {
            "available_datasets": {},
            "total_datasets": 0,
            "total_dataframes": 0,
            "configuration": {
                "geographic_level": self.geographic_level,
                "file_paths": list(self.file_paths) if hasattr(self.file_paths, '__iter__') else str(self.file_paths)
            },
            "data_type": "suitability_scores"
        }
        
        for level, data in self.suitability_scores.items():
            if data is not None:
                summary["total_datasets"] += 1
                
                if isinstance(data, pd.DataFrame):
                    summary["available_datasets"][level] = {
                        "type": "DataFrame",
                        "shape": data.shape,
                        "columns": list(data.columns) if len(data.columns) <= 10 else f"{len(data.columns)} columns"
                    }
                    summary["total_dataframes"] += 1
                else:
                    summary["available_datasets"][level] = {
                        "type": str(type(data)),
                        "note": "May not be saveable as CSV"
                    }
        
        return summary

# Example usage:
"""
# Basic usage - load all geographic levels
loader = SuitabilityScoreLoader(geographic_level="all")
suitability_scores = loader.load_suitability_scores()

# Save all loaded data to CSV files
loader.save_to_csv()

# Custom output directory
loader.save_to_csv(output_dir="custom_output/suitability")

# Save specific geographic levels only
loader.save_to_csv(geographic_levels=["county", "block_group"])

# Load specific geographic level
county_loader = SuitabilityScoreLoader(geographic_level="county")
county_scores = county_loader.load_suitability_scores()
county_loader.save_to_csv()

# Get summary of available data for saving
save_summary = loader.get_save_summary()
print(save_summary)

# Access individual datasets
county_data = loader.suitability_scores.get("county")
block_group_data = loader.suitability_scores.get("block_group")
project_data = loader.suitability_scores.get("project")

# File output structure:
# data_processed/suitability_scores/
# ├── county_suitability_scores.csv
# ├── block_group_suitability_scores.csv
# └── project_suitability_scores.csv
"""

