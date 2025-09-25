from utils.regression_utils import (
    reg_state_fixed
)

from constants.mappers import COVARIATE_MAPPING
from constants.regression_variables import REGRESSION_VARIABLES, COVARIATE_ORDER
from stargazer.stargazer import Stargazer


def run_base_regressions(data, x_variables):
    return {
        'y1': reg_state_fixed('y1', x_variables, data),
        'y2': reg_state_fixed('y2', x_variables, data),
        'y3': reg_state_fixed('y3', x_variables, data)
    }
    
    
def create_regression_table(models, custom_columns, dependent_variable_name=None):
    table = Stargazer(models)
    if custom_columns:
        table.custom_columns(custom_columns)
    if dependent_variable_name:
        table.dependent_variable_name(dependent_variable_name)
    # Filter COVARIATE_ORDER to only include variables that exist in the models
    present_covariates = set()
    for model in models:
        present_covariates.update(model.params.index)
    
    filtered_order = [var for var in COVARIATE_ORDER if var in present_covariates]
    table.covariate_order(filtered_order)
    table.rename_covariates(COVARIATE_MAPPING)
    
    return table
    