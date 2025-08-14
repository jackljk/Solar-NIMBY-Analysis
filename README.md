# Solar NIMBY Analysis

A comprehensive data analysis project examining solar energy suitability and social factors at county and block group levels across the United States.

## 📋 Project Overview

This repository contains tools and analysis for understanding the relationship between solar energy potential and various social, economic, and geographic factors. The project focuses on examining "Not In My Backyard" (NIMBY) attitudes and their correlation with renewable energy development.

## 🏗️ Project Structure

```
Solar-NIMBY-Analysis/
├── data/                           # Raw data files
│   ├── bounding_boxes/            # Geographic boundary data
│   ├── electric_data/             # Energy generation data
│   ├── mappings/                  # FIPS codes and geographic mappings
│   ├── raw_suitability_data/      # Raw suitability score TIF files
│   └── social_factors/            # Social and demographic data
├── data_processed/                # Processed data outputs
│   ├── county_level/              # County-level processed data
│   ├── block_group_level/         # Block group-level processed data
│   └── suitability_scores/       # Processed suitability scores
├── src/                           # Source code
│   ├── preprocess_data/           # Data processing modules
│   │   ├── county_level/          # County-level processors
│   │   ├── block_group_level/     # Block group-level processors
│   │   └── techno_econ_suitability_scores/  # Suitability analysis
│   ├── regression_analysis/       # Statistical analysis tools
│   ├── robustness_checks/         # Validation and robustness testing
│   └── visualizations/            # Data visualization notebooks
├── notebooks/                     # Analysis notebooks
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Required packages (install via pip or conda):
  ```bash
  pandas
  geopandas
  numpy
  rasterio
  rasterstats
  matplotlib
  plotly
  ipywidgets
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd Solar-NIMBY-Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # WIP - requirements file
   ```


## 📈 Analysis Components

### Data Types Processed

- **Demographic Data**: Race, education levels, unemployment rates
- **Economic Data**: Income levels, GDP, electricity prices
- **Energy Data**: Solar potential, wind data, existing installations
- **Geographic Data**: Land cover, protected areas, population density
- **Political Data**: Election results and voting patterns
- **Suitability Scores**: Technical and economic viability metrics

### Geographic Levels

- **County Level**: 3,000+ US counties
- **Block Group Level**: 200,000+ census block groups
- **Project Level**: Individual solar installation sites


