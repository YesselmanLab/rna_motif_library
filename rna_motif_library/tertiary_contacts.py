import concurrent.futures
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_motif_residues(motif_residues_csv_path: str) -> dict:
    """
    Load motif residues from a CSV file into a dictionary.

    Args:
        motif_residues_csv_path (str): Path to the CSV file containing motif residues.

    Returns:
        dict: A dictionary where keys are motif names and values are lists of residues.
    """
    with open(motif_residues_csv_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        motif_residues_dict = {}

        for row in csv_reader:
            name = row[0]  # Extract the motif name
            data = row[1:]  # Extract the residues
            motif_residues_dict[name] = data  # Store in the dictionary

    return motif_residues_dict


'''def find_tertiary_contacts(
        interactions_from_csv: pd.core.groupby.generic.DataFrameGroupBy,
        list_of_res_in_motifs: Dict[str, List[str]],
) -> None:
    """
    Find tertiary contacts from interaction data and write them to CSV files.

    Args:
        interactions_from_csv (DataFrameGroupBy): Grouped DataFrame of interactions.
        list_of_res_in_motifs (dict): Dictionary of residues in each motif.
    """
    f_tert = open("tertiary_contact_list.csv", "w")
    f_tert.write(
        "motif_1,motif_2,type_1,type_2,res_1,res_2,hairpin_len_1,hairpin_len_2,res_type_1,res_type_2"
        + "\n"
    )
    f_single = open("single_motif_inter_list.csv", "w")
    f_single.write(
        "motif,type_1,type_2,res_1,res_2,nt_1,nt_2,distance,angle,res_type_1,res_type_2"
        + "\n"
    )
    for interaction_group in interactions_from_csv:
        # interaction_group[0] is the name, [1] is the actual DF
        # HELIX.7PKQ.3.UGC-GCA.0 is the format; now you have the motif name as str
        name_of_source_motif = interaction_group[0]
        name_split = name_of_source_motif.split(".")
        source_motif_type = name_split[0]
        source_motif_cif_id = str(name_split[1])
        # get the residues for the motif of interest; look up in the dictionary
        residues_in_source_motif = list_of_res_in_motifs.get(
            name_of_source_motif
        )  # is a list of strings
        # now get the DF with the interaction data in it
        interaction_data_df = interaction_group[1]
        # iterate over each interaction in the motif
        for _, row in interaction_data_df.iterrows():
            # interaction data format: name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2
            # convert datatype series to list
            interaction_data = row.tolist()
            res_1 = interaction_data[1]
            res_2 = interaction_data[2]  # all are strings
            # only for f_single
            type_1 = interaction_data[3]
            type_2 = interaction_data[4]
            res_type_1 = interaction_data[11]
            res_type_2 = interaction_data[12]
            if len(type_1) == 1:
                nt_1 = "nt"
            else:
                nt_1 = "aa"
            if len(type_2) == 1:
                nt_2 = "nt"
            else:
                nt_2 = "aa"
            distance_data = str(interaction_data[7])
            angle_data = str(interaction_data[8])
            res_1_present = False
            res_2_present = False
            # need to check if the residue is present in ANY residue list where the CIF id is the same
            # so first we filter to get all the motifs in the same CIF
            dict_with_source_motif_PDB_motifs = {}
            first_line_skipped = False
            # filter mechanism
            for key, value in list_of_res_in_motifs.items():
                if not first_line_skipped:
                    first_line_skipped = True
                    continue
                if key.split(".")[1] == source_motif_cif_id:
                    dict_with_source_motif_PDB_motifs[key] = value
            # if either residue in the interaction is present in the source motif
            if res_1 in residues_in_source_motif:
                res_1_present = True
            if res_2 in residues_in_source_motif:
                res_2_present = True
            # check if residues are present, and if they are, handle them accordingly
            if (res_1_present == False) and (res_2_present == False):
                # not a tertiary contact, write to single_motif_inter_list.csv
                f_single.write(
                    name_of_source_motif
                    + ","
                    + type_1
                    + ","
                    + type_2
                    + ","
                    + res_1
                    + ","
                    + res_2
                    + ","
                    + nt_1
                    + ","
                    + nt_2
                    + ","
                    + distance_data
                    + ","
                    + angle_data
                    + ","
                    + res_type_1
                    + ","
                    + res_type_2
                    + "\n"
                )
            elif (res_1_present == True) and (res_2_present == True):
                # not a tert_contact
                pass
            elif (res_1_present == True) and (res_2_present == False):
                # tert contact found
                # res_1 is present in the current motif, res_2 is elsewhere so need to find it
                # now find which motif res_2 is in
                for (
                        motif_name,
                        motif_residue_list,
                ) in dict_with_source_motif_PDB_motifs.items():
                    # Check if the given string is present in the list
                    if res_2 in motif_residue_list:
                        print(motif_name)
                        motif_name_split = motif_name.split(".")
                        motif_name_type = motif_name_split[0]
                        # if the motifs are hairpins, get length
                        if motif_name_type == "HAIRPIN":
                            hairpin_length_1 = str(name_split[2])
                            hairpin_length_2 = str(motif_name_split[2])
                        else:
                            hairpin_length_1 = "0"
                            hairpin_length_2 = "0"
                        # print data to CSV
                        f_tert.write(
                            name_of_source_motif
                            + ","
                            + motif_name
                            + ","
                            + source_motif_type
                            + ","
                            + motif_name_type
                            + ","
                            + res_1
                            + ","
                            + res_2
                            + ","
                            + hairpin_length_1
                            + ","
                            + hairpin_length_2
                            + ","
                            + res_type_1
                            + ","
                            + res_type_2
                            + "\n"
                        )
            elif (res_1_present == False) and (res_2_present == True):
                # tert contact found
                # res_2 is present in the current motif, res_1 is elsewhere
                res_2_data = (res_2, name_of_source_motif)
                # now find which motif res_1 is in
                for (
                        motif_name,
                        motif_residue_list,
                ) in dict_with_source_motif_PDB_motifs.items():
                    # Check if the given string is present in the list
                    if res_1 in motif_residue_list:
                        print(motif_name)
                        motif_name_split = motif_name.split(".")
                        motif_name_type = motif_name_split[0]
                        # if the motifs are hairpins, get length
                        if motif_name_type == "HAIRPIN":
                            hairpin_length_1 = str(name_split[2])
                            hairpin_length_2 = str(motif_name_split[2])
                        else:
                            hairpin_length_1 = "0"
                            hairpin_length_2 = "0"
                        # print data to CSV (only if the motif names are the same)
                        f_tert.write(
                            name_of_source_motif
                            + ","
                            + motif_name
                            + ","
                            + source_motif_type
                            + ","
                            + motif_name_type
                            + ","
                            + res_1
                            + ","
                            + res_2
                            + ","
                            + hairpin_length_1
                            + ","
                            + hairpin_length_2
                            + ","
                            + res_type_1
                            + ","
                            + res_type_2
                            + "\n"
                        )
'''

def find_tertiary_contacts(interactions_from_csv, list_of_res_in_motifs, threads):
    """
    Multithreaded function to find tertiary contacts from interaction data and write them to CSV files.

    Args:
        interactions_from_csv: DataFrameGroupBy object of interactions grouped by motif name.
        list_of_res_in_motifs: Dictionary of motif names to list of residues in each motif.
        threads: Number of threads to use for parallel processing.
    """
    # Opening files outside the thread pool to avoid issues with concurrent writes
    with open("tertiary_contact_list.csv", "w") as f_tert, open("single_motif_inter_list.csv", "w") as f_single:
        # Write headers
        f_tert.write("motif_1,motif_2,type_1,type_2,res_1,res_2,hairpin_len_1,hairpin_len_2,res_type_1,res_type_2\n")
        f_single.write("motif,type_1,type_2,res_1,res_2,nt_1,nt_2,distance,angle,res_type_1,res_type_2\n")

        def process_interaction_group(interaction_group):
            name_of_source_motif, interaction_data_df = interaction_group
            # Extract motif and CIF ID from the motif name
            name_split = name_of_source_motif.split(".")
            source_motif_type = name_split[0]
            source_motif_cif_id = name_split[1]
            residues_in_source_motif = list_of_res_in_motifs.get(name_of_source_motif, [])
            dict_with_source_motif_PDB_motifs = {
                key: value for key, value in list_of_res_in_motifs.items() if key.split(".")[1] == source_motif_cif_id
            }

            outputs = []

            for _, row in interaction_data_df.iterrows():
                res_1, res_2 = row['res_1'], row['res_2']
                type_1, type_2 = row['type_1'], row['type_2']
                res_type_1, res_type_2 = row['res_type_1'], row['res_type_2']
                nt_1, nt_2 = "nt" if len(type_1) == 1 else "aa", "nt" if len(type_2) == 1 else "aa"
                distance, angle = str(row['distance']), str(row['angle'])

                if res_1 in residues_in_source_motif and res_2 in residues_in_source_motif:
                    # Both residues are in the same motif, skip
                    continue
                elif res_1 in residues_in_source_motif or res_2 in residues_in_source_motif:
                    # One of the residues is in the motif, tertiary contact
                    outputs.append(f"{name_of_source_motif},{type_1},{type_2},{res_1},{res_2},{nt_1},{nt_2},{distance},{angle},{res_type_1},{res_type_2}\n")
                else:
                    # Neither of the residues is in the motif, single motif interaction
                    f_single.write(f"{name_of_source_motif},{type_1},{type_2},{res_1},{res_2},{nt_1},{nt_2},{distance},{angle},{res_type_1},{res_type_2}\n")

            return outputs

        # Create a thread pool and process groups concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            results = executor.map(process_interaction_group, interactions_from_csv)

        # Write all results to the tertiary contact list file
        for result in results:
            for line in result:
                f_tert.write(line)



def find_unique_tertiary_contacts():
    """
    Finds unique tertiary contacts from the CSV file and writes them to another CSV file.
    """
    # after the CSV for tertiary contacts are made we need to go through and extract all unique pairs in CSV
    tert_contact_csv_df = pd.read_csv(
        "tertiary_contact_list.csv"
    )  # used to make unique list which then is used for everything else

    # Check if the required columns are present
    required_columns = ["motif_1", "motif_2"]
    if not set(required_columns).issubset(tert_contact_csv_df.columns):
        print(
            f"A line in the CSV is blank. If this shows only once, it is working as intended."
        )

    print("Finding unique tertiary contacts...")
    # Count unique tert contacts and print to a CSV
    grouped_unique_tert_contacts = tert_contact_csv_df.groupby(
        ["motif_1", "motif_2", "res_1", "res_2"]
    )

    # Specify the file path
    # Making the unique_tert_contacts.csv file
    print("Opened unique_tert_contacts.csv")
    csv_file_path = "unique_tert_contacts.csv"
    print("Writing to unique_tert_contacts.csv...")
    with open(csv_file_path, mode="w", newline="") as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write the header
        writer.writerow(
            [
                "motif_1",
                "motif_2",
                "seq_1",
                "seq_2",
                "type_1",
                "type_2",
                "res_1",
                "res_2",
                "count",
                "res_type_1",
                "res_type_2",
            ]
        )

        # Iterate over groups
        for group_name, group_df in grouped_unique_tert_contacts:
            # Get the count of rows in the group
            count = len(group_df)

            # Drop duplicates
            group_df_unique = group_df.drop_duplicates(keep="first")

            # Iterate over rows to write to CSV
            for index, row in group_df_unique.iterrows():
                # Split motif_1 and motif_2 by "." and take all parts except the last one
                seq_1 = ".".join(row["motif_1"].split(".")[:-1])
                seq_2 = ".".join(row["motif_2"].split(".")[:-1])

                # need to write the motif type correctly or you get fuckups later (sometimes they get switched up)
                motif_1_spl = row["motif_1"].split(".")
                motif_2_spl = row["motif_2"].split(".")

                motif_1_type = motif_1_spl[0]
                motif_2_type = motif_2_spl[0]

                # Write to CSV, appending the count at the end
                writer.writerow(
                    [
                        row["motif_1"],
                        row["motif_2"],
                        seq_1,
                        seq_2,
                        motif_1_type,
                        motif_2_type,
                        row["res_1"],
                        row["res_2"],
                        count,
                        row["res_type_1"],
                        row["res_type_2"],
                    ]
                )
    # Close the file
    file.close()


def delete_duplicate_contacts() -> pd.DataFrame:
    """
    Deletes duplicate tertiary contacts where motif_1 and motif_2 are switched and processes the data to remove further duplicates.

    Returns:
        pd.DataFrame: A DataFrame of unique tertiary contacts.
    """
    # graph hydrogen bonds per overall tertiary contact
    unique_tert_contact_df_new = pd.read_csv("unique_tert_contacts.csv")

    print("Deleting duplicates...")
    # Now delete duplicate interactions (where motif_1 and 2 are switched)
    # Sort the 'motif_1' and 'motif_2' columns within each row
    unique_tert_contact_df_new[["motif_1", "motif_2"]] = pd.DataFrame(
        np.sort(unique_tert_contact_df_new[["motif_1", "motif_2"]], axis=1),
        index=unique_tert_contact_df_new.index,
    )

    # Group the DataFrame by the sorted 'motif_1' and 'motif_2' columns
    grouped_df = unique_tert_contact_df_new.groupby(["motif_1", "motif_2"])
    # Apply the function to remove duplicate rows within each group
    unique_tert_contact_df_for_hbonds = grouped_df.apply(
        remove_duplicate_res
    ).reset_index(drop=True)
    unique_tert_contact_df_for_hbonds.to_csv(
        "unique_tert_contacts_for_hbonds.csv", index=False
    )
    # If there are fewer than two residues interacting in the contact, delete
    grouped_df = unique_tert_contact_df_for_hbonds.groupby(["motif_1", "motif_2"])
    # Filter out groups with less than 2 rows
    unique_tert_contact_df_for_hbonds = grouped_df.filter(lambda x: len(x) >= 2)

    # Print it for good measure, debug
    # unique_tert_contact_df_for_hbonds.to_csv("unique_tert_contacts.csv", index=False)

    # Process each group to sum the 'count' column and replace values
    final_data = []
    grouped_df = unique_tert_contact_df_for_hbonds.groupby(["motif_1", "motif_2"])
    for name, group in grouped_df:
        # Sum the 'count' column within the group
        count_sum = group["count"].astype(int).sum()

        # Set the 'count' column to the summed value
        group["count"] = count_sum

        # Keep only the first row of each group
        final_data.append(group.iloc[0])

    # Create a new DataFrame from the processed data
    unique_tert_contact_df = pd.DataFrame(final_data)

    # Remove duplicates
    # Group the DataFrame by the sorted 'seq_1' and 'seq_2' columns
    grouped_df = unique_tert_contact_df.groupby(["seq_1", "seq_2"])

    # get rid of duplicate sequences
    unique_tert_contact_df = grouped_df.first().reset_index()

    # Reset the index of the DataFrame
    unique_tert_contact_df.reset_index(drop=True, inplace=True)

    # Print it for good measure
    unique_tert_contact_df.to_csv("unique_tert_contacts.csv", index=False)

    return unique_tert_contact_df


def remove_duplicate_res(group):
    """
    Remove duplicate rows within each group based on sorted 'res_1' and 'res_2' values.
    Args:
        group (pd.DataFrame): DataFrame group to process.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    # Sort 'res_1' and 'res_2' columns within each row
    group[["res_1", "res_2"]] = group[["res_1", "res_2"]].apply(sort_res, axis=1)
    # Drop duplicate rows based on sorted 'res_1' and 'res_2' values
    return group.drop_duplicates(subset=["res_1", "res_2"], keep="first")


def sort_res(row: pd.Series) -> pd.Series:
    """
    Sort 'res_1' and 'res_2' columns within each row.
    Args:
        row (pd.Series): Series containing 'res_1' and 'res_2' values.
    Returns:
        pd.Series: Series with sorted 'res_1' and 'res_2' values.
    """
    return pd.Series(np.sort(row.values))


def print_tert_contacts_to_csv(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Print tertiary contacts to CSV files.
    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.
    """
    # make directory for tert contacts
    __safe_mkdir("tertiary_contacts")
    print("Saving tertiary contacts to CIF files...")
    # for printing the tert contact CIFs need to prepare data
    motifs_1 = unique_tert_contact_df["motif_1"].tolist()
    motifs_2 = unique_tert_contact_df["motif_2"].tolist()
    types_1 = unique_tert_contact_df["type_1"].tolist()
    types_2 = unique_tert_contact_df["type_2"].tolist()
    ress_1 = unique_tert_contact_df["res_1"].tolist()
    ress_2 = unique_tert_contact_df["res_2"].tolist()
    # Create a list of tuples
    motif_pairs = [
        (motif1, motif2, types1, types2, ress1, ress2)
        for motif1, motif2, types1, types2, ress1, ress2 in zip(
            motifs_1, motifs_2, types_1, types_2, ress_1, ress_2
        )
    ]
    # Create a list of tuples with the third element specifying the count
    unique_motif_pairs_with_count = [
        (pair[0], pair[1], pair[2], pair[3], pair[4], pair[5])
        for pair in set(motif_pairs)
    ]
    for motif_pair in unique_motif_pairs_with_count:
        motif_1 = motif_pair[0]
        motif_2 = motif_pair[1]
        # if motif_1 or motif_2 are hairpins less than 3 NTS long then don't process
        motif_1_split = motif_1.split(".")
        motif_2_split = motif_2.split(".")
        motif_1_name = motif_1_split[0]
        motif_2_name = motif_2_split[0]
        try:
            motif_1_hairpin_len = float(motif_1_split[2])
            motif_2_hairpin_len = float(motif_2_split[2])
        except ValueError:
            motif_1_hairpin_len = 0
            motif_2_hairpin_len = 0
        if not (
                (motif_1_name == "HAIRPIN" or motif_2_name == "HAIRPIN")
                and ((0 < motif_1_hairpin_len < 3) or (0 < motif_2_hairpin_len < 3))
        ):
            directory_to_search = "motifs"
            motif_cif_1 = str(motif_1) + ".cif"
            motif_cif_2 = str(motif_2) + ".cif"
            path_to_cif_1 = find_cif_file(directory_to_search, motif_cif_1)
            path_to_cif_2 = find_cif_file(directory_to_search, motif_cif_2)
            # path
            tert_contact_name = motif_1 + "." + motif_2
            # classifying them based on motif type
            motif_1_type = motif_1.split(".")[0]
            motif_2_type = motif_2.split(".")[0]
            motif_types_list = [motif_1_type, motif_2_type]
            motif_types_sorted = sorted(motif_types_list)
            motif_types = str(motif_types_sorted[0]) + "-" + str(motif_types_sorted[1])

            if motif_types:
                __safe_mkdir("tertiary_contacts/" + motif_types)
                tert_contact_out_path = (
                        "tertiary_contacts/" + motif_types + "/" + tert_contact_name
                )
            else:
                tert_contact_out_path = "tertiary_contacts/" + tert_contact_name
            print(tert_contact_name)
            # take the CIF files and merge them
            try:
                merge_cif_files(
                    file1_path=path_to_cif_1,
                    file2_path=path_to_cif_2,
                    output_path=f"{tert_contact_out_path}.cif",
                    lines_to_delete=24,
                )
            except TypeError:
                continue
        else:
            continue


def __safe_mkdir(directory: str) -> None:
    """Safely creates a directory if it does not already exist.
    Args:
        directory: The path of the directory to create.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


# merges the contents of CIF files
def merge_cif_files(
        file1_path: str, file2_path: str, output_path: str, lines_to_delete: int
) -> None:
    """
    Merge the contents of two CIF files into one.

    Args:
        file1_path (str): Path to the first CIF file.
        file2_path (str): Path to the second CIF file.
        output_path (str): Path to the output CIF file.
        lines_to_delete (int): Number of lines to delete from the start of the second CIF file.
    """
    # Read the contents of the first CIF file
    with open(file1_path, "r") as file1:
        content_file1 = file1.readlines()

    # Read the contents of the second CIF file
    with open(file2_path, "r") as file2:
        content_file2 = file2.readlines()

    # Delete the first x lines from the second CIF file
    content_file2 = content_file2[lines_to_delete:]

    # Combine the contents of the first and modified second CIF files
    merged_content = content_file1 + content_file2

    # Write the merged content to the output CIF file
    with open(output_path, "w") as output_file:
        output_file.writelines(merged_content)


# finds CIF files inside the given directory plus all subdirectories
def find_cif_file(directory_path: str, file_name: str) -> Optional[str]:
    """
    Search for a CIF file in a directory and its subdirectories.

    Args:
        directory_path (str): The path of the directory to search.
        file_name (str): The name of the CIF file to find.

    Returns:
        Optional[str]: The full path to the found CIF file, or None if not found.
    """
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Check if the file is in the current directory
        if file_name in files:
            # Return the full path to the file
            return os.path.join(root, file_name)

    # If the file is not found
    return None


def plot_tert_histograms(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Plot histograms of tertiary contacts.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.
    """
    print("Plotting...")
    unique_tert_contact_df_for_hbonds = pd.read_csv(
        "unique_tert_contacts_for_hbonds.csv"
    )

    # Group by motif_1 and motif_2 and sum the counts; to determine how many h-bonds there are between tert contacts
    hbond_counts_in_terts = (
        unique_tert_contact_df_for_hbonds.groupby(["motif_1", "motif_2"])["count"]
        .sum()
        .reset_index()
    )

    # Rename the 'count' column to 'sum_hbonds'
    hbond_counts_in_terts.rename(columns={"count": "sum_hbonds"}, inplace=True)
    # Remove duplicate lines based on motif_1 and motif_2 columns
    hbond_counts_in_terts.drop_duplicates(subset=["motif_1", "motif_2"], inplace=True)
    # Plot settings
    tick_positions = np.arange(
        hbond_counts_in_terts["sum_hbonds"].min(),
        hbond_counts_in_terts["sum_hbonds"].max() + 1,
    )

    sns.set_theme(style="white")  # palette='deep', color_codes=True)
    plt.rcParams.update({"font.size": 20})  # Set overall text size

    plot_hbonds_per_tert(
        hbond_counts_in_terts=hbond_counts_in_terts, tick_positions=tick_positions
    )
    plot_hairpins_in_tert(unique_tert_contact_df=unique_tert_contact_df)
    plot_helices_in_tert(unique_tert_contact_df=unique_tert_contact_df)
    plot_sstrands_in_tert(unique_tert_contact_df=unique_tert_contact_df)


def plot_sstrands_in_tert(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Plot the lengths of single strands (sstrands) in tertiary contacts.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.
    """
    # sstrands in tertiary contacts
    # filter to get only sstrands
    sstrand_cols_1 = ["motif_1", "type_1", "res_1", "seq_1"]
    sstrand_tert_contact_df_1 = unique_tert_contact_df[sstrand_cols_1]
    sstrand_cols_2 = ["motif_2", "type_2", "res_2", "seq_2"]
    sstrand_tert_contact_df_2 = unique_tert_contact_df[sstrand_cols_2]
    # Filter rows where types are equal to "SSTRAND"
    sstrand_tert_contact_df_1 = sstrand_tert_contact_df_1[
        sstrand_tert_contact_df_1["type_1"] == "SSTRAND"
        ]
    sstrand_tert_contact_df_2 = sstrand_tert_contact_df_2[
        sstrand_tert_contact_df_2["type_2"] == "SSTRAND"
        ]
    # split
    split_column_1 = sstrand_tert_contact_df_1["motif_1"].str.split(".")
    split_column_2 = sstrand_tert_contact_df_2["motif_2"].str.split(".")
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    sstrand_tert_contact_df_1 = sstrand_tert_contact_df_1.assign(length_1=length_1)
    sstrand_tert_contact_df_2 = sstrand_tert_contact_df_2.assign(length_2=length_2)
    # Concatenate and drop dupes
    new_tert_df = pd.concat(
        [sstrand_tert_contact_df_1, sstrand_tert_contact_df_2],
        ignore_index=True,
        axis=0,
    )
    new_tert_df.drop_duplicates(subset=["seq_1"], keep="first", inplace=True)
    # List of column names to delete
    columns_to_delete = ["motif_2", "type_2", "res_2", "seq_2", "length_2"]
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    # rename columns
    new_tert_df.columns = ["motif", "type", "res", "seq", "sstrand_length"]
    for index, row in new_tert_df.iterrows():
        seq_value = row["seq"]
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            sstrand_length_value = int(parts[2])
            new_tert_df.at[index, "sstrand_length"] = sstrand_length_value

    # Print for debug
    new_tert_df.to_csv("sstrand_tert.csv", index=False)
    # Convert 'sstrand_length' column to numeric type
    new_tert_df["sstrand_length"] = pd.to_numeric(
        new_tert_df["sstrand_length"], errors="coerce"
    )
    tick_positions = np.arange(
        new_tert_df["sstrand_length"].min(), new_tert_df["sstrand_length"].max() + 1
    )

    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(
        new_tert_df["sstrand_length"],
        bins=np.arange(
            new_tert_df["sstrand_length"].min() - 0.5,
            new_tert_df["sstrand_length"].max() + 1.5,
            1,
        ),
        edgecolor="black",
        width=0.8,
    )  # adjust bins as needed
    plt.xlabel("Length of sstrands in tertiary contacts")
    plt.ylabel("Count")
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5)

    # Save the plot as PNG file
    plt.savefig("figure_3_sstrand_in_tert.png", dpi=600)
    # Close the plot
    plt.close()


def plot_helices_in_tert(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Plot the lengths of helices in tertiary contacts.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.
    """
    # filter to get only helices
    helix_cols_1 = ["motif_1", "type_1", "res_1", "seq_1"]
    helix_tert_contact_df_1 = unique_tert_contact_df[helix_cols_1]
    helix_cols_2 = ["motif_2", "type_2", "res_2", "seq_2"]
    helix_tert_contact_df_2 = unique_tert_contact_df[helix_cols_2]
    # Filter rows where types are equal to "HELIX"
    helix_tert_contact_df_1 = helix_tert_contact_df_1[
        helix_tert_contact_df_1["type_1"] == "HELIX"
        ]
    helix_tert_contact_df_2 = helix_tert_contact_df_2[
        helix_tert_contact_df_2["type_2"] == "HELIX"
        ]
    # split
    split_column_1 = helix_tert_contact_df_1["motif_1"].str.split(".")
    split_column_2 = helix_tert_contact_df_2["motif_2"].str.split(".")
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    helix_tert_contact_df_1 = helix_tert_contact_df_1.assign(length_1=length_1)
    helix_tert_contact_df_2 = helix_tert_contact_df_2.assign(length_2=length_2)
    # concatenate and get rid of dupes
    new_tert_df = pd.concat(
        [helix_tert_contact_df_1, helix_tert_contact_df_2], ignore_index=True, axis=0
    )
    new_tert_df.drop_duplicates(subset=["seq_1"], keep="first", inplace=True)
    # List of column names to delete
    columns_to_delete = ["motif_2", "type_2", "res_2", "seq_2", "length_2"]
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    new_tert_df.columns = ["motif", "type", "res", "seq", "helix_length"]

    for index, row in new_tert_df.iterrows():
        seq_value = row["seq"]
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            helix_length_value = int(parts[2])
            new_tert_df.at[index, "helix_length"] = helix_length_value

    # Print for debug
    new_tert_df.to_csv("helices_tert.csv", index=False)
    # Convert 'helix_length' column to numeric type
    new_tert_df["helix_length"] = pd.to_numeric(
        new_tert_df["helix_length"], errors="coerce"
    )
    tick_positions = np.arange(
        new_tert_df["helix_length"].min(), new_tert_df["helix_length"].max() + 1
    )
    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(
        new_tert_df["helix_length"],
        bins=np.arange(
            new_tert_df["helix_length"].min() - 0.5,
            new_tert_df["helix_length"].max() + 1.5,
            1,
        ),
        edgecolor="black",
        width=0.8,
    )  # adjust bins as needed
    plt.xlabel("Length of helices in tertiary contacts")
    plt.ylabel("Count")
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5)

    # Save the plot as PNG file
    plt.savefig("figure_3_helices_in_tert.png", dpi=600)
    # Close the plot
    plt.close()


def plot_hairpins_in_tert(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Plot the lengths of hairpins in tertiary contacts.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.
    """
    # Now make a histogram for lengths of hairpins in tertiary contacts
    # split into two DFs
    df_cols_1 = ["motif_1", "type_1", "res_1", "seq_1"]
    tert_contact_df_1 = unique_tert_contact_df[df_cols_1]
    df_cols_2 = ["motif_2", "type_2", "res_2", "seq_2"]
    tert_contact_df_2 = unique_tert_contact_df[df_cols_2]
    # Filter rows where hairpins_1 and hairpins_2 are equal to "HAIRPIN"
    tert_contact_df_1 = tert_contact_df_1[tert_contact_df_1["type_1"] == "HAIRPIN"]
    tert_contact_df_2 = tert_contact_df_2[tert_contact_df_2["type_2"] == "HAIRPIN"]
    # split
    split_column_1 = tert_contact_df_1["motif_1"].str.split(".")
    split_column_2 = tert_contact_df_2["motif_2"].str.split(".")
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    tert_contact_df_1 = tert_contact_df_1.assign(length_1=length_1)
    tert_contact_df_2 = tert_contact_df_1.assign(length_2=length_2)
    # Concatenate tert_contact_df_1 and tert_contact_df_2
    new_tert_df = pd.concat(
        [tert_contact_df_1, tert_contact_df_2], ignore_index=True, axis=0
    )
    # List of column names to delete (duplicates, since the were concatenated one on top of another)
    columns_to_delete = ["length_2"]
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    # And delete dupes
    new_tert_df.drop_duplicates(subset=["seq_1"], keep="first", inplace=True)
    # Rename columns of tert_contact_df_1
    new_tert_df.columns = ["motif", "type", "res", "seq", "hairpin_length"]
    for index, row in new_tert_df.iterrows():
        seq_value = row["seq"]
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            hairpin_length_value = int(parts[2])
            new_tert_df.at[index, "hairpin_length"] = hairpin_length_value

    # Print for debug reasons
    new_tert_df.to_csv("hairpins_tert.csv", index=False)
    # Convert data to numeric from string
    new_tert_df["hairpin_length"] = pd.to_numeric(
        new_tert_df["hairpin_length"], errors="coerce"
    )
    # Set tick positions to fit the range of data
    tick_positions = np.arange(
        new_tert_df["hairpin_length"].min(), new_tert_df["hairpin_length"].max() + 1
    )
    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(
        new_tert_df["hairpin_length"],
        bins=np.arange(
            new_tert_df["hairpin_length"].min() - 0.5,
            new_tert_df["hairpin_length"].max() + 1.5,
            1,
        ),
        edgecolor="black",
        width=0.8,
    )  # adjust bins as needed
    plt.xlabel("Length of hairpins in tertiary contacts")
    plt.ylabel("Count")
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5); old code
    # Save the plot as PNG file
    plt.savefig("figure_3_hairpins_in_tert.png", dpi=600)
    # Close the plot
    plt.close()


def plot_hbonds_per_tert(
        hbond_counts_in_terts: pd.DataFrame, tick_positions: np.ndarray
) -> None:
    """
    Plot a histogram of the number of hydrogen bonds per tertiary contact.

    Args:
        hbond_counts_in_terts (pd.DataFrame): DataFrame containing counts of hydrogen bonds in tertiary contacts.
        tick_positions (np.ndarray): Array of tick positions for the x-axis.
    """
    # Plot histogram
    # H-bonds per tert, need to group the ones with like motifs and sum the tert contacts
    plt.figure(figsize=(6, 6))
    plt.hist(
        hbond_counts_in_terts["sum_hbonds"],
        bins=np.arange(
            hbond_counts_in_terts["sum_hbonds"].min() + 0.5,
            hbond_counts_in_terts["sum_hbonds"].max() + 1.5,
            1,
        ),
        edgecolor="black",
        width=0.8,
    )  # adjust bins as needed
    plt.xlabel("H-bonds per tertiary contact")
    plt.ylabel("Count")
    # Set ticks to start at 2 and step every 5 values
    adjusted_tick_positions = np.arange(2, hbond_counts_in_terts["sum_hbonds"].max() + 1, 5)
    plt.xticks(adjusted_tick_positions, [str(tick) for tick in adjusted_tick_positions])
    # Add tick marks on x-axis
    # plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # Save the plot as PNG file
    plt.savefig("figure_3_hbonds_per_tert.png", dpi=600)
    # Close the plot
    plt.close()
