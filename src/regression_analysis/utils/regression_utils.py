import pandas as pd
import statsmodels.api as sm


from stargazer.stargazer import Stargazer

import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from stargazer.stargazer import Stargazer

def reg_state_fixed(y_variable, x_col, data):
    """Regression with state fixed effects."""
    # Replace spaces in state names with underscores for formula compatibility
    cleaned_data = data.copy()
    # make it take either state or State
    if 'state' not in cleaned_data.columns and 'State' in cleaned_data.columns:
        cleaned_data = cleaned_data.rename(columns={'State': 'state'})
        
    cleaned_data['state'] = cleaned_data['state'].str.replace(' ', '_')

    # Create dummy variables for each state (excluding reference state)
    state_dummy_vars = pd.get_dummies(cleaned_data['state'], prefix='state')
    state_dummy_vars = state_dummy_vars.drop('state_Colorado', axis=1)  # Colorado as reference

    # Combine original data with state dummies
    regression_data = pd.concat([cleaned_data, state_dummy_vars], axis=1)

    # Build regression formula with state fixed effects
    state_terms = '+'.join(state_dummy_vars.columns)
    x_terms = '+'.join(x_col.columns)
    regression_formula = f"{y_variable} ~ 1 + {state_terms} + {x_terms}"
    
    # Fit OLS model
    fitted_model = sm.OLS.from_formula(regression_formula, regression_data).fit()

    # Prepare results for Stargazer output
    model_results = [fitted_model]
    stargazer_output = Stargazer(model_results)

    return {
        'stargazer': stargazer_output,
        'model': fitted_model
    }


def reg_county_fixed(y_variable, x_col, data):
    """Regression with county fixed effects."""
    # Prepare data with clean state names and county identifiers
    cleaned_data = data.copy()
    cleaned_data['state'] = cleaned_data['state'].str.replace(' ', '_')
    cleaned_data['state_county'] = cleaned_data['state'] + "_" + cleaned_data['county']

    # Build regression formula using categorical county fixed effects
    x_terms = '+'.join(x_col.columns)
    regression_formula = f"{y_variable} ~ 1 + C(state_county) + {x_terms}"

    # Fit OLS model with county fixed effects
    fitted_model = sm.OLS.from_formula(regression_formula, cleaned_data).fit()

    # Prepare results for Stargazer output
    stargazer_output = Stargazer([fitted_model])

    return stargazer_output, fitted_model

def multiple_regression_intern(y_variable, x_col, data, interacting_var):
    """Multiple regression with state-specific interaction terms."""
    # Replace spaces in state names with underscores for formula compatibility
    cleaned_data = data.copy()
    cleaned_data['state'] = cleaned_data['state'].str.replace(' ', '_')

    # Create dummy variables for each state (excluding reference state)
    state_dummy_vars = pd.get_dummies(cleaned_data['state'], prefix='state')
    state_dummy_vars = state_dummy_vars.drop('state_Colorado', axis=1)  # Colorado as reference

    # Combine original data with state dummies
    regression_data = pd.concat([cleaned_data, state_dummy_vars], axis=1)

    # Create interaction terms between state dummies and the interacting variable
    interaction_terms = [f"{state_col}*{interacting_var}" for state_col in state_dummy_vars.columns]
    
    # Build regression formula with main effects and interactions
    x_terms = '+'.join(x_col.columns)
    interaction_terms_str = ' + '.join(interaction_terms)
    regression_formula = f"{y_variable} ~ 1 + {x_terms} + {interaction_terms_str}"

    # Fit OLS model
    fitted_model = sm.OLS.from_formula(regression_formula, regression_data).fit()

    # Prepare results for Stargazer output
    model_results = [fitted_model]
    stargazer_output = Stargazer(model_results)

    return stargazer_output, fitted_model


def multiple_regression(y_variable, x_col, data):
    """Simple multiple regression without fixed effects."""
    # Build regression formula with intercept and independent variables
    x_terms = '+'.join(x_col.columns)
    regression_formula = f"{y_variable} ~ 1 + {x_terms}"
    
    # Fit OLS model
    fitted_model = sm.OLS.from_formula(regression_formula, data).fit()

    # Prepare results for Stargazer output
    model_results = [fitted_model]
    stargazer_output = Stargazer(model_results)

    return stargazer_output, fitted_model


def base_social_fixed(y_variable, x_col, data, social):
    """Regression with state fixed effects including social variables."""
    # Replace spaces in state names with underscores for formula compatibility
    cleaned_data = data.copy()
    cleaned_data['state'] = cleaned_data['state'].str.replace(' ', '_')

    # Create dummy variables for each state (excluding reference state)
    state_dummy_vars = pd.get_dummies(cleaned_data['state'], prefix='state')
    state_dummy_vars = state_dummy_vars.drop('state_Colorado', axis=1)  # Colorado as reference

    # Combine original data with state dummies
    regression_data = pd.concat([cleaned_data, state_dummy_vars], axis=1)

    # Build regression formula with state fixed effects, main variables, and social variables
    state_terms = '+'.join(state_dummy_vars.columns)
    x_terms = '+'.join(x_col.columns)
    social_terms = '+'.join(social.columns)
    regression_formula = f"{y_variable} ~ 1 + {state_terms} + {x_terms} + {social_terms}"
    
    # Fit OLS model
    fitted_model = sm.OLS.from_formula(regression_formula, regression_data).fit()

    # Prepare results for Stargazer output
    model_results = [fitted_model]
    stargazer_output = Stargazer(model_results)

    return stargazer_output, fitted_model


def probit(y_variable, x_col, data):
    """Probit regression with state fixed effects."""
    # Replace spaces in state names with underscores for compatibility
    cleaned_data = data.copy()
    cleaned_data['state'] = cleaned_data['state'].str.replace(' ', '_')

    # Create dummy variables for each state (excluding reference state)
    state_dummy_vars = pd.get_dummies(cleaned_data['state'], prefix='state')
    state_dummy_vars = state_dummy_vars.drop('state_Colorado', axis=1)  # Colorado as reference

    # Combine original data with state dummies
    regression_data = pd.concat([cleaned_data, state_dummy_vars], axis=1)

    # Prepare independent variables matrix and dependent variable
    independent_vars = pd.concat([regression_data[state_dummy_vars.columns], x_col], axis=1)
    independent_vars_with_constant = sm.add_constant(independent_vars)  # Add intercept
    dependent_var = regression_data[y_variable]

    # Fit probit regression model
    fitted_probit_model = Probit(dependent_var, independent_vars_with_constant).fit()

    # Prepare results for Stargazer output
    model_results = [fitted_probit_model]
    stargazer_output = Stargazer(model_results)

    return stargazer_output, fitted_probit_model