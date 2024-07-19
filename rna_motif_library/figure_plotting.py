import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_present_hbonds(grouped_hbond_df):
    for group_name, hbonds in grouped_hbond_df:
        # type_1, type_2, atom_1, atom_2 = map(str, group_name)
        (res_1, atom_1), (res_2, atom_2) = group_name
        type_1, type_2 = res_1, res_2

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        hbonds_subset = hbonds[["distance", "angle"]].reset_index(drop=True)

        if len(hbonds_subset) >= 100:  # if there are more than 100 data points
            map_name = f"{type_1}-{type_2} {atom_1}-{atom_2}"

            heatmap_csv_path = "data/out_csvs/heatmap_data"
            __safe_mkdir(heatmap_csv_path)

            heat_data_csv_path = f"{heatmap_csv_path}/{map_name}.csv"
            hbonds.to_csv(heat_data_csv_path, index=False)

        else:
            print(
                f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient data points."
            )


def plot_present_hbonds(grouped_hbond_df):
    # Process each group to make heatmaps; code above is optimized and this is the OG version
    # OG version makes more sense after cleaning it up
    for group_name, hbonds in grouped_hbond_df:
        # type_1, type_2, atom_1, atom_2 = map(str, group_name)
        (res_1, atom_1), (res_2, atom_2) = group_name
        type_1, type_2 = res_1, res_2

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        hbonds_subset = hbonds[["distance", "angle"]].reset_index(drop=True)

        if len(hbonds_subset) >= 100:  # if there are more than 100 data points
            # Set global font size (was set up there)
            # plt.rc('font', size=18)

            distance_bins = np.arange(
                2.0, 4.1, 0.1
            )  # Bins from 2 to 4 in increments of 0.1
            angle_bins = np.arange(0, 181, 10)  # Bins from 0 to 180 in increments of 10

            hbonds_subset["distance_bin"] = pd.cut(
                hbonds_subset["distance"], bins=distance_bins
            )
            hbonds_subset["angle_bin"] = pd.cut(hbonds_subset["angle"], bins=angle_bins)

            heatmap_data = (
                hbonds_subset.groupby(["angle_bin", "distance_bin"])
                .size()
                .unstack(fill_value=0)
            )

            plt.figure(figsize=(6, 6))
            sns.heatmap(
                heatmap_data,
                cmap="gray_r",
                xticklabels=1,
                yticklabels=range(0, 181, 10),
                square=True,
            )

            plt.xticks(
                np.arange(len(distance_bins)) + 0.5,
                [f"{bin_val:.1f}" for bin_val in distance_bins],
                rotation=0,
            )
            plt.yticks(np.arange(len(angle_bins)) + 0.5, angle_bins, rotation=0)

            plt.xlabel("Distance (Å)")
            plt.ylabel("Angle (°)")
            map_name = f"{type_1}-{type_2} {atom_1}-{atom_2}"
            # plt.title(f"{map_name} H-bond heatmap")

            map_dir = (
                "heatmaps/RNA-RNA"
                if len(type_1) == 1 and len(type_2) == 1
                else "heatmaps/RNA-PROT"
            )
            __safe_mkdir(map_dir)

            map_path = f"{map_dir}/{map_name}.png"
            plt.savefig(map_path, dpi=250)
            plt.close()

            heatmap_csv_path = "heatmap_data"
            __safe_mkdir(heatmap_csv_path)

            heat_data_csv_path = f"{heatmap_csv_path}/{map_name}.csv"
            hbonds.to_csv(heat_data_csv_path, index=False)

            # 2D histogram
            plt.figure(figsize=(6, 4.8))
            plt.hist2d(
                hbonds_subset["distance"],
                hbonds_subset["angle"],
                bins=[distance_bins, angle_bins],
                cmap="gray_r",
            )
            plt.xlabel("Distance (Å)")
            plt.ylabel("Angle (°)")
            plt.colorbar(label="Count")
            plt.title(f"{map_name} H-bond heatmap")

            plt.savefig(map_path, dpi=250)
            plt.close()

        else:
            print(
                f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient data points."
            )


def __safe_mkdir(directory: str) -> None:
    """Safely creates a directory if it does not already exist.

    Args:
        directory: The path of the directory to create.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
