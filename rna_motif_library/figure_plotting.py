import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


"""def plot_twoway_size_heatmap(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Check if there is any data in the DataFrame
        if df.empty:
            print("No data in the CSV file. Skipping twoway junction processing.")
            return
    except pd.errors.EmptyDataError:
        print(
            "EmptyDataError: No data in the CSV file regarding twoway junctions. Skipping twoway junction processing."
        )
        return
    # make a heatmap for TWOWAY JCTs
    # Create a DataFrame for the heatmap
    df["bridging_nts_0"] = df["bridging_nts_0"] - 2
    df["bridging_nts_1"] = df["bridging_nts_1"] - 2
    twoway_heatmap_df = df.pivot_table(
        index="bridging_nts_0", columns="bridging_nts_1", aggfunc="size", fill_value=0
    )

    # Extract the data from the DataFrame
    x = twoway_heatmap_df.columns.astype(float)
    y = twoway_heatmap_df.index.astype(float)
    z = twoway_heatmap_df.values

    # Reshape the data for hist2d
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Determine the range of x and y
    x_range = np.arange(
        int(x.min()), min(int(x.max()) + 1, 12)
    )  # Limit to 10 on x-axis
    y_range = np.arange(
        int(y.min()), min(int(y.max()) + 1, 12)
    )  # Limit to 10 on y-axis

    # Create the 2D histogram
    sns.set_theme(style="white")
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 20})
    heatmap = plt.hist2d(
        x_mesh.ravel(),
        y_mesh.ravel(),
        weights=z.ravel(),
        bins=[x_range, y_range],
        cmap="gray_r",
    )

    # Add labels and title
    plt.xlabel("Strand 1 Nucleotides")
    plt.ylabel("Strand 2 Nucleotides")
    # plt.title("Figure 2(e): 2-way junctions (X-Y)", fontsize=32)

    # Add colorbar for frequency scale
    # cbar = plt.colorbar(label='Frequency')

    # Set aspect ratio of color bar to match the height of the plot
    # cbar.ax.set_aspect(40)

    # Set ticks on x-axis
    plt.xticks(
        np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1),
        [
            f"{int(tick - 0.5)}"
            for tick in np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1)
        ],
    )

    # Set ticks on y-axis
    plt.yticks(
        np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1),
        [
            f"{int(tick - 0.5)}"
            for tick in np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1)
        ],
    )

    # Set aspect ratio to square
    plt.gca().set_aspect("equal", adjustable="box")

    # Add colorbar for frequency scale
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(heatmap[3], cax=cax)
    cbar.set_label("Count")
    # cbar.ax.tick_params(labelsize=18)

    # Save the heatmap as a PNG file
    plt.savefig("figure_2_twoway_motif_heatmap.png", dpi=600)

    # Don't display the plot
    plt.close()
"""


def plot_twoway_size_heatmap(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("No data in the CSV file. Skipping twoway junction processing.")
            return
    except pd.errors.EmptyDataError:
        print(
            "EmptyDataError: No data in the CSV file regarding twoway junctions. Skipping twoway junction processing."
        )
        return

    df["bridging_nts_0"] = df["bridging_nts_0"] - 2
    df["bridging_nts_1"] = df["bridging_nts_1"] - 2
    twoway_heatmap_df = df.pivot_table(
        index="bridging_nts_0", columns="bridging_nts_1", aggfunc="size", fill_value=0
    )

    x = twoway_heatmap_df.columns.astype(float)
    y = twoway_heatmap_df.index.astype(float)
    z = twoway_heatmap_df.values
    x_mesh, y_mesh = np.meshgrid(x, y)

    x_range = np.arange(int(x.min()), min(int(x.max()) + 1, 12))
    y_range = np.arange(int(y.min()), min(int(y.max()) + 1, 12))

    sns.set_theme(style="white")
    plt.figure(figsize=(7, 6))
    plt.rcParams.update({"font.size": 20})
    heatmap = plt.hist2d(
        x_mesh.ravel(),
        y_mesh.ravel(),
        weights=z.ravel(),
        bins=[x_range, y_range],
        cmap="gray_r",
    )

    plt.xlabel("Strand 1 Nucleotides")
    plt.ylabel("Strand 2 Nucleotides")

    plt.xticks(
        np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1),
        [
            f"{int(tick - 0.5)}"
            for tick in np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1)
        ],
    )
    plt.yticks(
        np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1),
        [
            f"{int(tick - 0.5)}"
            for tick in np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1)
        ],
    )

    plt.gca().set_aspect("equal", adjustable="box")
    # Adjust margins
    plt.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.06)

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(heatmap[3], cax=cax)
    cbar.set_label("Count")

    plt.savefig("figure_2_twoway_motif_heatmap.png", dpi=600)
    plt.close()


def plot_tert_contact_type_counts(tert_contact_csv_directory):
    # Load the CSV into a DataFrame and keep only the columns 'res_type_1' and 'res_type_2'
    df = pd.read_csv(tert_contact_csv_directory, usecols=["res_type_1", "res_type_2"])
    df = df.replace({"res_type_1": "aa", "res_type_2": "aa"}, "base")

    tuples_list = [tuple(sorted(x)) for x in df.to_records(index=False)]

    # Count the occurrences of each tuple
    tuple_counts = Counter(tuples_list)

    # Convert the counts to a DataFrame for easier plotting
    tuple_counts_df = pd.DataFrame(
        tuple_counts.items(), columns=["Contact Type", "Count"]
    )

    # Sort the DataFrame by Count (optional)
    tuple_counts_df = tuple_counts_df.sort_values(by="Count")

    # Plot the bar graph
    plt.figure(figsize=(6, 6))
    plt.barh(
        tuple_counts_df["Contact Type"].astype(str),
        tuple_counts_df["Count"],
        edgecolor="black",
        height=0.8,
        color=sns.color_palette()[0],
    )
    plt.xlabel("Count")
    plt.ylabel("Tertiary Contact Type")
    plt.title("")  # tertiary contact types
    plt.tight_layout()

    # Save the graph as a PNG file
    plt.savefig("figure_3_residue_contact_types.png", dpi=600)
    plt.close()


def plot_tert_contact_counts(tert_motif_directory):
    # Create a dictionary to store counts for each folder
    tert_folder_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(tert_motif_directory):
        item_path = os.path.join(tert_motif_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            tert_folder_counts[item_name] = file_count
    # make a bar graph of all types of motifs
    tert_folder_names = list(tert_folder_counts.keys())
    tert_file_counts = list(tert_folder_counts.values())
    # Sort the folder names and file counts alphabetically
    tert_folder_names_sorted, tert_file_counts_sorted = zip(
        *sorted(zip(tert_folder_names, tert_file_counts))
    )
    plt.figure(figsize=(6, 6))
    plt.barh(
        tert_folder_names_sorted,
        tert_file_counts_sorted,
        edgecolor="black",
        color=sns.color_palette()[0],
        height=0.8,
    )
    plt.xlabel("Count")
    plt.ylabel("Tertiary Contact Type")
    plt.title("")  # tertiary contact types
    # Adjust x-axis ticks for a tight fit
    # plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    # Save the graph as a PNG file
    plt.savefig("figure_3_tertiary_motif_counts.png", dpi=600)
    # Don't display the plot
    plt.close()


def plot_sstrand_counts(motif_directory):
    # Of the single strands, how long are they (bar graph)
    sstrand_directory = motif_directory + "/sstrand"

    sstrand_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(sstrand_directory):
        item_path = os.path.join(sstrand_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            sstrand_counts[item_name] = file_count
    # Convert helix folder names to integers and sort them
    sorted_sstrand_counts = dict(
        sorted(sstrand_counts.items(), key=lambda item: int(item[0]))
    )
    # Extract sorted keys and values
    sstrand_folder_names_sorted = list(sorted_sstrand_counts.keys())
    sstrand_file_counts_sorted = list(sorted_sstrand_counts.values())
    # Convert helix folder names to integers
    sstrand_bins = sorted([int(name) for name in sstrand_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(sstrand_bins), max(sstrand_bins) + 1)
    plt.figure(figsize=(6, 6))
    plt.hist(
        sstrand_bins,
        bins=np.arange(min(sstrand_bins) - 0.5, max(sstrand_bins) + 1.5, 1),
        weights=sstrand_file_counts_sorted,
        color=sns.color_palette()[0],
        width=0.8,
        edgecolor="black",
        align="mid",
    )
    plt.xlabel("Single Strand Length")
    plt.ylabel("Count")
    plt.title("")  # Helices with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(
        np.arange(min(sstrand_bins), max(sstrand_bins) + 1, 5)
    )  # Display ticks every 5 integers
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig("figure_2_sstrand_counts_bar_graph.png", dpi=600)
    # Don't display the plot
    plt.close()


def plot_helix_counts(motif_directory):
    # of the helices, how long are they (bar graph)
    helix_directory = motif_directory + "/helices"

    helix_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(helix_directory):
        item_path = os.path.join(helix_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            helix_counts[item_name] = file_count
    # Convert helix folder names to integers and sort them
    sorted_helix_counts = dict(
        sorted(helix_counts.items(), key=lambda item: int(item[0]))
    )
    # Extract sorted keys and values
    helix_folder_names_sorted = list(sorted_helix_counts.keys())
    helix_file_counts_sorted = list(sorted_helix_counts.values())
    # Convert helix folder names to integers
    helix_bins = sorted([int(name) for name in helix_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(helix_bins), max(helix_bins) + 1)
    plt.figure(figsize=(6, 6))
    plt.hist(
        helix_bins,
        bins=np.arange(min(helix_bins) - 0.5, max(helix_bins) + 1.5, 1),
        weights=helix_file_counts_sorted,
        color=sns.color_palette()[0],
        edgecolor="black",
        align="mid",
        width=0.8,
    )
    plt.xlabel("Helix Length")
    plt.ylabel("Count")
    plt.title("")  # Helices with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(
        np.arange(min(helix_bins), max(helix_bins) + 1, 5)
    )  # Display ticks every 5 integers
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig("figure_2_helix_counts_bar_graph.png", dpi=600)
    # Don't display the plot
    plt.close()


def plot_motif_counts(motif_directory):
    print("Plotting...")
    # graphs
    # Create a dictionary to store counts for each folder
    folder_counts = {
        "TWOWAY": 0,
        "NWAY": 0,
        "HAIRPIN": 0,
        "HELIX": 0,
        "SSTRAND": 0,
    }  # Initialize counts

    # for folder in directory, count numbers:
    # try:
    # Iterate over all items in the specified directory
    for item_name in os.listdir(motif_directory):
        item_path = os.path.join(motif_directory, item_name)

        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Check if the folder name is "2ways"
            if item_name == "2ways":
                # If folder name is "2ways", register the count as TWOWAY
                folder_counts["TWOWAY"] += file_count
            elif "ways" in item_name:
                # If folder name contains "ways" but is not "2ways", register the count as NWAY
                folder_counts["NWAY"] += file_count
            elif item_name == "hairpins":
                # If folder name is "hairpins", register the count as HAIRPIN
                folder_counts["HAIRPIN"] += file_count
            elif item_name == "helices":
                # If folder name is "helices", register the count as HELIX
                folder_counts["HELIX"] += file_count
            elif item_name == "sstrand":
                # If folder name is "sstrand", register count as SSTRAND
                folder_counts["SSTRAND"] += file_count
            else:
                # If the folder name doesn't match any condition, use it as is
                folder_counts[item_name] = file_count

    # Convert the dictionary to a DataFrame
    data = pd.DataFrame(list(folder_counts.items()), columns=["Motif Type", "Count"])
    # Sort the DataFrame by 'Motif Type'
    data = data.sort_values("Motif Type")
    # Set the Seaborn theme
    sns.set_theme(style="white")  # palette='deep', color_codes=True)
    # Create the bar plot
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.rcParams.update({"font.size": 20})  # Set overall text size
    sns.barplot(
        data=data,
        x="Motif Type",
        y="Count",
        color=sns.color_palette()[0],
        edgecolor="black",
    )
    # Set labels and title
    plt.xlabel("Motif Type")
    plt.ylabel("Count")
    plt.title("")
    # Adjust layout for better fit
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig("figure_2_bar_graph_motif_counts.png", dpi=600)
    # Close the plot to avoid display
    plt.close()


def plot_hairpin_counts(motif_directory):
    # of the hairpins, how long are they (histogram)
    hairpin_directory = motif_directory + "/hairpins"
    hairpin_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(hairpin_directory):
        item_path = os.path.join(hairpin_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            hairpin_counts[item_name] = file_count
    # Convert hairpin folder names to integers and sort them
    sorted_hairpin_counts = dict(
        sorted(hairpin_counts.items(), key=lambda item: int(item[0]))
    )
    # Extract sorted keys and values
    hairpin_folder_names_sorted = list(sorted_hairpin_counts.keys())
    hairpin_file_counts_sorted = list(sorted_hairpin_counts.values())
    # Convert hairpin folder names to integers
    hairpin_bins = sorted([int(name) for name in hairpin_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(hairpin_bins), max(hairpin_bins) + 1)
    sns.set_theme(style="white")
    plt.figure(figsize=(6, 6))
    plt.hist(
        hairpin_bins,
        bins=np.arange(min(hairpin_bins) - 0.5, max(hairpin_bins) + 1.5, 1),
        weights=hairpin_file_counts_sorted,
        color=sns.color_palette()[0],
        edgecolor="black",
        align="mid",
        width=0.8,
    )
    plt.xlabel("Hairpin Length")
    plt.ylabel("Count")
    plt.title("")  # Hairpins with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(
        np.arange(min(hairpin_bins), max(hairpin_bins) + 1, 5)
    )  # Display ticks every 5 integers
    # plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig("figure_2_hairpin_counts_bar_graph.png", dpi=600)
    # Don't display the plot
    plt.close()


# counts all files with a specific extension (used in generating figures)
def count_files_with_extension(directory_path, file_extension):
    try:
        # Initialize a counter
        file_count = 0

        # Iterate over the directory and its subdirectories
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                # Check if the current file has the specified extension
                if filename.endswith(file_extension):
                    file_count += 1

        return file_count
    except Exception as e:
        print(f"Error counting files in directory '{directory_path}': {e}")
        return None


def __safe_mkdir(directory: str) -> None:
    """Safely creates a directory if it does not already exist.

    Args:
        directory: The path of the directory to create.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
