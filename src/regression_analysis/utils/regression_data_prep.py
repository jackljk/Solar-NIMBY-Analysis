from typing import Optional
from .common import normalize_data, unnormalize_by_area


def prepare_regression_data(data, relevant_columns, columns_to_normalize, columns_to_unnormalized: Optional[dict[str, str]] = None, drop_subset: Optional[list[str]] = None):

    cols = [col for sublist in relevant_columns.values() for col in sublist]
    prep_data = data[cols]
    # normalize columns
    prep_data = normalize_data(prep_data, columns_to_normalize)
    
    if drop_subset:
        prep_data = prep_data.dropna(subset=drop_subset)

    # set NaNs to 0
    prep_data = prep_data.fillna(0)

    # unnormalize columns if specified
    if columns_to_unnormalized:
        for col, area_col in columns_to_unnormalized.items():
            prep_data = unnormalize_by_area(prep_data, col, area_col)

    return prep_data