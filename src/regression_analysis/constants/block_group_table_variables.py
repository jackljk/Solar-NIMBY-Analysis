class VariableRegistry:
    """Central registry for all variable definitions used in regression analysis."""
    
    # Basic geographic columns
    BG_BASIC_COLUMNS = ["State Name", "County Name", "GEOID"]
    
    # Demographic variables
    DEMOGRAPHIC_COLUMNS = [
        "Unemployment Rate",
        "White Only",
        "African American Only",
        "Asian Only",
        "Median Household Income",
    ]
    
    # Political variables
    POLITICAL_COLUMNS = [
        "Democratic Vote Percentage",
        "Republican Vote Percentage",
    ]
    
    # Education variables
    EDUCATION_COLUMNS = [
        "less_than_9th_grade",
        "grade_9th_to_12th_no_diploma", 
        "high_school_graduate",
        "some_college_no_degree",
        "associate_degree",
        "bachelor_degree",
        "graduate_degree",
    ]
    
    # Wind energy variables
    WIND_COLUMNS = [
        "Wind Capacity Intensity (MW/ 1000 sq mile)",
        "Wind Project Intensity (Projects/ 1000 sq mile)",
    ]
    
    # Solar energy variables (base)
    SOLAR_BASE_COLUMNS = [
        "Solar Capacity Intensity MW_per_mi2",
        "Solar Project Intensity Count_per_mi2",
        "Solar Avg Capacity Intensity MW_per_mi2",
    ]
    
    
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
    
    @classmethod
    def get_social_factors_columns(cls) -> list[str]:
        """Get all social factor columns."""
        return (cls.DEMOGRAPHIC_COLUMNS + 
                cls.POLITICAL_COLUMNS + 
                cls.EDUCATION_COLUMNS)

# List of factors that are not in normalized scale, so we normalize them for all regression analysis
COLS_TO_NORMALIZE = (
    VariableRegistry.DEMOGRAPHIC_COLUMNS +
    VariableRegistry.POLITICAL_COLUMNS +
    ["25+ Less than 9th grade", "25+ High school graduate", "25+ Bachelor's degree", "25+ Graduate or professional degree"] +
    VariableRegistry.WIND_COLUMNS
)

# =============================================================================
# COLUMN BUILDER FUNCTIONS Block Group
# =============================================================================

def build_base_bg_columns() -> dict[str, list[str]]:
    """Build the base block group column configuration."""
    return {
        "basic": VariableRegistry.BG_BASIC_COLUMNS,
        "Social Factors": VariableRegistry.get_social_factors_columns(),
        "Solar": VariableRegistry.SOLAR_BASE_COLUMNS,
        "Wind": VariableRegistry.WIND_COLUMNS,
        "suitability": VariableRegistry.SUITABILITY_BASE_COLUMNS,
    }

# =============================================================================
# COLUMN CONFIGURATIONS Block Group (Generated from builders)
# =============================================================================

BASE_BG_COLUMNS = build_base_bg_columns()
    