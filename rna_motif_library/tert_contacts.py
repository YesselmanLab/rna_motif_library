import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rna_motif_library.figure_plotting import safe_mkdir


def load_motif_residues(motif_residues_csv_path: str) -> dict:
    """
    Load motif residues from a CSV file into a dictionary.

    Args:
        motif_residues_csv_path (str): Path to the CSV file containing motif residues.

    Returns:
        motif_residues_dict (dict): A dictionary where keys are motif names and values are lists of residues.

    """
    with open(motif_residues_csv_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        motif_residues_dict = {}

        for row in csv_reader:
            name = row[0]  # Extract the motif name
            data = row[1:]  # Extract the residues
            motif_residues_dict[name] = data  # Store in the dictionary

    return motif_residues_dict


def find_tertiary_contacts(
    interactions_from_csv: pd.core.groupby.generic.DataFrameGroupBy,
    list_of_res_in_motifs: Dict[str, List[str]],
    csv_dir: str,
) -> None:
    """
    Find tertiary contacts from interaction data and write them to CSV files.

    Args:
        interactions_from_csv (pd.DataFrameGroupBy): Grouped DataFrame of interactions.
        list_of_res_in_motifs (dict): Dictionary of residues in each motif.
        csv_dir (str): Directory of CSV results.

    Returns:
        None

    """
    f_tert = open(os.path.join(csv_dir, "tertiary_contact_list.csv"), "w")
    f_tert.write(
        "motif_1,motif_2,type_1,type_2,res_1,res_2,hairpin_len_1,hairpin_len_2,res_type_1,res_type_2"
        + "\n"
    )
    f_single = open(os.path.join(csv_dir, "single_motif_inter_list.csv"), "w")
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


def find_unique_tertiary_contacts(csv_dir: str) -> None:
    """
    Finds unique tertiary contacts from the CSV file and writes them to another CSV file.

    Args:
        csv_dir (str): Directory of CSV results

    Returns:
        None

    """
    # after the CSV for tertiary contacts are made we need to go through and extract all unique pairs in CSV
    tert_contact_csv_df = pd.read_csv(
        os.path.join(csv_dir, "tertiary_contact_list.csv")
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
    csv_file_path = os.path.join(csv_dir, "unique_tert_contacts.csv")
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


def delete_duplicate_contacts(csv_dir: str) -> pd.DataFrame:
    """
    Deletes duplicate tertiary contacts where motif_1 and motif_2 are switched and processes the data to remove further duplicates.

    Args:
        csv_dir (str): Directory of CSV results

    Returns:
        unique_tert_contact_df (pd.DataFrame): A DataFrame of unique tertiary contacts.

    """
    # graph hydrogen bonds per overall tertiary contact
    unique_tert_contact_df_new = pd.read_csv(
        os.path.join(csv_dir, "unique_tert_contacts.csv")
    )

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
        os.path.join(csv_dir, "unique_tert_contacts_for_hbonds.csv"), index=False
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
    unique_tert_contact_df.to_csv(
        os.path.join(csv_dir, "unique_tert_contacts.csv"), index=False
    )

    return unique_tert_contact_df


def remove_duplicate_res(group: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows within each group based on sorted 'res_1' and 'res_2' values.

    Args:
        group: DataFrame group to process.

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


def print_tert_contacts_to_cif(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Print tertiary contacts to CIF files.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.

    Returns:
        None

    """
    # make directory for tert contacts
    safe_mkdir("data/tertiary_contacts")
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
            directory_to_search = "data/motifs"
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
                safe_mkdir("data/tertiary_contacts/" + motif_types)
                tert_contact_out_path = (
                    "data/tertiary_contacts/" + motif_types + "/" + tert_contact_name
                )
            else:
                tert_contact_out_path = "data/tertiary_contacts/" + tert_contact_name
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
        lines_to_delete (str): Number of lines to delete from the start of the second CIF file.

    Returns:
        None

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
        file_path (str): The full path to the found CIF file, or None if not found.

    """
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Check if the file is in the current directory
        if file_name in files:
            # Return the full path to the file
            return os.path.join(root, file_name)

    # If the file is not found
    return None
