SHORT_TO_LONG_MAPPER = {
    # Solar Suitability Variables
    "X1": "GHI",
    "X2": "Protected_Land",
    "X3": "Habitat",
    "X4": "Slope",
    "X5": "Population_Density",
    "X6": "Distance_to_Substation",
    "X7": "Land_Cover",
    "Intercept": "const",
    # Solar Variables
    "y1": "Solar Avg Capacity Intensity (MW/ 1000 sq mile)",
    "y2": "Solar Capacity Intensity (MW/ 1000 sq mile)",
    "y3": "Solar Project Intensity (Projects/ 1000 sq mile)",
    "y1_small": "Solar Avg Capacity Intensity (MW/ 1000 sq mile)_small",
    "y2_small": "Solar Capacity Intensity (MW/ 1000 sq mile)_small",
    "y3_small": "Solar Project Intensity (Projects/ 1000 sq mile)_small",
    "y1_med": "Solar Avg Capacity Intensity (MW/ 1000 sq mile)_medium",
    "y2_med": "Solar Capacity Intensity (MW/ 1000 sq mile)_medium",
    "y3_med": "Solar Project Intensity (Projects/ 1000 sq mile)_medium",
    "y1_lg": "Solar Avg Capacity Intensity (MW/ 1000 sq mile)_large",
    "y2_lg": "Solar Capacity Intensity (MW/ 1000 sq mile)_large",
    "y3_lg": "Solar Project Intensity (Projects/ 1000 sq mile)_large",
    # Wind Variables
    "wind_intens": "Wind Capacity Intensity (MW/ 1000 sq mile)",
    "wind_proj": "Wind Project Intensity (Projects/ 1000 sq mile)",
    # Social Variables
    "unemp_rate": "Unemployment Rate",
    "hispanic": "Hispanic/Latino",
    "white": "White",
    "black": "Black/African American",
    "asian": "Asian",
    "dem_vote": "democrat_percentage_vote",
    "repub_vote": "republican_percentage_vote",
    "less9": "25+ Less than 9th grade",
    "9to12": "25+ 9th to 12th grade, no diploma",
    "HS": "25+ High school graduate",
    "some_college": "25+ Some college, no degree",
    "associates": "25+ Associate's degree",
    "bach": "25+ Bachelor's degree",
    "grad": "25+ Graduate or professional degree",
    "income000": "Median Income",
    "rural": "Rural Area Percentage",
    "urban": "Urban Area Percentage",
    "resi": "Electric Residential Rate",
    "comm": "Electric Commercial Rate",
    "industry": "Electric Industrial Rate",
    "GDPpercapita": "GDP_2022",
    "roof_capacity": "Number of Existing Installs (Projects/ 1000 sq mile)"
}
LONG_TO_SHORT_MAPPER = {v: k for k, v in SHORT_TO_LONG_MAPPER.items()}


COVARIATE_MAPPING = {
    # Solar Suitability Variables
    "X1": "GHI",
    "X2": "Unprotected Land",
    "X3": "Habitat",
    "X4": "Slope",
    "X5": "Population Sparsity",
    "X7": "Land Cover",
    "Intercept": "const",
    # Social Variables
    "resi": 'Residential',
    "comm": 'Commercial',
    "industry": 'Industrial',
    "income000": 'Income',
    "unemp_rate": 'Unemployment',
    "hispanic": 'Hispanic',
    "white": 'White',
    "black": 'Black',
    "asian": 'Asian',
    "dem_vote": 'Democrat',
    "repub_vote": 'Republican',
    "HS": 'HS',
    "grad": 'Grad',
    # Wind Variables
    "wind_intens": "Wind Intensity",
}
