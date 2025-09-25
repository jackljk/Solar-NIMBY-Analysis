# =============================================================================
# BASE VARIABLE REGISTRY
# =============================================================================

class VariableRegistry:
    """Central registry for all variable definitions used in regression analysis."""
    
    # Basic geographic columns
    BASIC_COLUMNS = ["State", "area mi2"]
    COUNTY_BASIC_COLUMNS = ["State", "County Name", "area mi2"]
    BG_BASIC_COLUMNS = ["State", "County Name", "GEOID"]
    
    # Demographic variables
    DEMOGRAPHIC_COLUMNS = [
        "Unemployment Rate",
        "Hispanic/Latino", 
        "White",
        "Black/African American",
        "Asian",
        "Median Income",
    ]
    
    # Political variables
    POLITICAL_COLUMNS = [
        "democrat_percentage_vote",
        "republican_percentage_vote",
    ]
    
    # Education variables
    EDUCATION_COLUMNS = [
        "25+ Less than 9th grade",
        "25+ 9th to 12th grade, no diploma", 
        "25+ High school graduate",
        "25+ Some college, no degree",
        "25+ Associate's degree",
        "25+ Bachelor's degree",
        "25+ Graduate or professional degree",
    ]
    
    # Land use variables
    LAND_USE_COLUMNS = [
        "Rural Area Percentage",
        "Urban Area Percentage",
    ]
    
    # Energy pricing variables
    ELECTRIC_RATE_COLUMNS = [
        "Electric Commercial Rate",
        "Electric Industrial Rate", 
        "Electric Residential Rate",
    ]
    
    # Wind energy variables
    WIND_COLUMNS = [
        "Wind Capacity Intensity (MW/ 1000 sq mile)",
        "Wind Project Intensity (Projects/ 1000 sq mile)",
    ]
    
    # Solar energy variables (base)
    SOLAR_BASE_COLUMNS = [
        "Solar Capacity Intensity (MW/ 1000 sq mile)",
        "Solar Project Intensity (Projects/ 1000 sq mile)",
        "Solar Avg Capacity Intensity (MW/ 1000 sq mile)",
    ]
    
    # Solar energy variables by size category
    SOLAR_SIZE_SUFFIXES = ["_small", "_medium", "_large"]
    
    # Suitability score variables
    SUITABILITY_BASE_COLUMNS = [
        "GHI",
        "Protected_Land",
        "Habitat", 
        "Slope",
        "Population_Density",
        "Land_Cover",
    ]
    
    # Extended suitability (includes distance to substation)
    SUITABILITY_EXTENDED_COLUMNS = SUITABILITY_BASE_COLUMNS + ["Distance_to_Substation"]
    
    # Economic variables
    ECONOMIC_COLUMNS = ["GDP_2022"]
    
    # Rooftop solar variables
    ROOFTOP_COLUMNS = ["Number of Existing Installs (Projects/ 1000 sq mile)"]

    @classmethod
    def get_solar_columns_with_sizes(cls) -> list[str]:
        """Generate all solar columns including size categories."""
        columns = cls.SOLAR_BASE_COLUMNS.copy()
        for base_col in cls.SOLAR_BASE_COLUMNS:
            for suffix in cls.SOLAR_SIZE_SUFFIXES:
                columns.append(f"{base_col}{suffix}")
        return columns

    @classmethod
    def get_social_factors_columns(cls) -> list[str]:
        """Get all social factor columns."""
        return (cls.DEMOGRAPHIC_COLUMNS + 
                cls.POLITICAL_COLUMNS + 
                cls.EDUCATION_COLUMNS)
    
    @classmethod
    def get_social_factors_with_land_use(cls) -> list[str]:
        """Get social factors including land use variables."""
        return cls.get_social_factors_columns() + cls.LAND_USE_COLUMNS

# List of factors that are not in normalized scale, so we normalize them for all regression analysis
COLS_TO_NORMALIZE = (
    VariableRegistry.DEMOGRAPHIC_COLUMNS +
    VariableRegistry.POLITICAL_COLUMNS +
    ["25+ Less than 9th grade", "25+ High school graduate", "25+ Bachelor's degree", "25+ Graduate or professional degree"] +
    VariableRegistry.WIND_COLUMNS +
    VariableRegistry.LAND_USE_COLUMNS +
    VariableRegistry.ELECTRIC_RATE_COLUMNS
)

# =============================================================================
# COLUMN BUILDER FUNCTIONS County
# =============================================================================

def build_base_social_factors_columns() -> dict[str, list[str]]:
    """Build the base social factors column configuration."""
    return {
        "basic": VariableRegistry.BASIC_COLUMNS,
        "Social Factors": VariableRegistry.get_social_factors_columns(),
        "Solar": VariableRegistry.get_solar_columns_with_sizes(),
        "Wind": VariableRegistry.WIND_COLUMNS,
        "suitability": VariableRegistry.SUITABILITY_BASE_COLUMNS,
    }

def build_urban_columns() -> dict[str, list[str]]:
    """Build urban analysis column configuration."""
    return {
        "basic": VariableRegistry.COUNTY_BASIC_COLUMNS,
        "Wind": VariableRegistry.WIND_COLUMNS,
        "Solar": VariableRegistry.get_solar_columns_with_sizes(),
        "Social Factors": VariableRegistry.get_social_factors_with_land_use(),
        "suitability": VariableRegistry.SUITABILITY_EXTENDED_COLUMNS,
    }

def build_electric_price_columns() -> dict[str, list[str]]:
    """Build electric price analysis column configuration."""
    return {
        "basic": VariableRegistry.COUNTY_BASIC_COLUMNS,
        "Wind": VariableRegistry.WIND_COLUMNS,
        "Solar": VariableRegistry.get_solar_columns_with_sizes(),
        "Social Factors": VariableRegistry.get_social_factors_columns() + VariableRegistry.ELECTRIC_RATE_COLUMNS,
        "suitability": VariableRegistry.SUITABILITY_EXTENDED_COLUMNS,
    }

def build_gdp_columns() -> dict[str, list[str]]:
    """Build GDP analysis column configuration."""
    return {
        "basic": VariableRegistry.COUNTY_BASIC_COLUMNS,
        "Wind": VariableRegistry.WIND_COLUMNS,
        "Solar": VariableRegistry.get_solar_columns_with_sizes(),
        "Social Factors": VariableRegistry.get_social_factors_with_land_use() + VariableRegistry.ECONOMIC_COLUMNS,
        "suitability": VariableRegistry.SUITABILITY_BASE_COLUMNS,
    }

def build_rooftop_columns() -> dict[str, list[str]]:
    """Build rooftop solar analysis column configuration."""
    return {
        "basic": VariableRegistry.COUNTY_BASIC_COLUMNS,
        "Wind": VariableRegistry.WIND_COLUMNS,
        "Solar": VariableRegistry.get_solar_columns_with_sizes(),
        "Social Factors": VariableRegistry.get_social_factors_with_land_use() + VariableRegistry.ROOFTOP_COLUMNS,
        "suitability": VariableRegistry.SUITABILITY_BASE_COLUMNS,
    }


# =============================================================================
# COLUMN CONFIGURATIONS COUNTY (Generated from builders)
# =============================================================================

BASE_SOCIAL_FACTORS_COLUMNS = build_base_social_factors_columns()

URBAN_COLUMNS = build_urban_columns()

ELECTRIC_PRICE_COLUMNS = build_electric_price_columns()

GDP_COLUMNS = build_gdp_columns()

ROOFTOP_COLUMNS = build_rooftop_columns()

