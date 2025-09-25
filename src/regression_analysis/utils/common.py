import pandas as pd

def normalize_data(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        max_value = df[col].max()
        df.loc[:, col] = df[col].apply(lambda x: x / max_value * 100 if pd.notnull(x) else x)
    return df


def unnormalize_by_area(df, columns, area_column):
    assert area_column in df.columns, f"Area column '{area_column}' not found in DataFrame."
    for col in columns:
        if col not in df.columns:
            continue
        # add as a new column with suffix '_unnormalized'
        new_col_name = f"{col}_unnormalized"
        df.loc[:, new_col_name] = df.apply(lambda row: row[col] * row[area_column] if pd.notnull(row[col]) and pd.notnull(row[area_column]) else row[col], axis=1)
    return df


def list_remove_value(lst, val):
    if val in lst:
        lst.remove(val)
    return lst

def render_and_save_table(stargazer_obj, file_path):
    """Renders a Stargazer object to LaTeX and saves it to a file."""
    with open(file_path, 'w') as f:
        f.write(stargazer_obj.render_latex())