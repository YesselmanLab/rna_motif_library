import os
import pandas as pd


def save_present_hbonds(grouped_hbond_df: pd.core.groupby.DataFrameGroupBy) -> None:
    """
    Saves groups of H-bonding data into CSVs.

    Args:
        grouped_hbond_df (pd.core.groupby.DataFrameGroupBy): dataframe containing H-bonding data

    Returns:
        None

    """
    for group_name, hbonds in grouped_hbond_df:
        (res_1, atom_1), (res_2, atom_2) = group_name[0]
        type_1, type_2 = res_1, res_2

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        map_name = f"{type_1}-{type_2} {atom_1}-{atom_2}"

        heatmap_csv_path = "data/out_csvs/heatmap_data"
        safe_mkdir(heatmap_csv_path)

        heat_data_csv_path = f"{heatmap_csv_path}/{map_name}.csv"
        hbonds.to_csv(heat_data_csv_path, index=False)


def safe_mkdir(directory: str) -> None:
    """
    Safely creates a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.

    Returns:
        None

    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
