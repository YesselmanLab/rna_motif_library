import csv
import os
from typing import Dict, List, Optional

import pandas as pd

from rna_motif_library.dssr_hbonds import assign_res_type


def find_unique_tert_contacts(tert_contact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters potential tertiary contacts to return only uniques.

    Args:
        tert_contact_df (pd.DataFrame): DataFrame containing potential tertiary contacts.
    Returns:
        unique_tert_contact_df (pd.DataFrame): DataFrame with only unique tertiary contacts.

    """
    # Create a new column with motifs sorted alphabetically
    tert_contact_df["sorted_motifs"] = tert_contact_df.apply(
        lambda row: tuple(sorted([row["motif_1"], row["motif_2"]])), axis=1
    )

    # Group by this new column and keep only the first line from each group
    grouped = tert_contact_df.groupby("sorted_motifs").first().reset_index()

    # Drop the 'sorted_motifs' column, since it's no longer needed
    unique_tert_contact_df = grouped.drop(columns=["sorted_motifs"])

    # Now delete those tertiary contacts with like sequences
    # First need to extract sequences by removing the dupe indicator
    def dupe_remover(column):
        """Function to process motifs by removing the dupe indicator thus keeping the sequence"""
        return column.str.split(".").str[:-1].str.join(".")

    unique_tert_contact_df["seq_1"] = dupe_remover(unique_tert_contact_df["motif_1"])
    unique_tert_contact_df["seq_2"] = dupe_remover(unique_tert_contact_df["motif_2"])
    # Now remove dupes according to sequence
    # Create a new column with sequences sorted alphabetically
    unique_tert_contact_df["sorted_seqs"] = unique_tert_contact_df.apply(
        lambda row: tuple(sorted([row["seq_1"], row["seq_2"]])), axis=1
    )

    # Group by the sorted sequences and keep only the first line from each group
    final_grouped = unique_tert_contact_df.groupby("sorted_seqs").first().reset_index()

    # Drop the 'sorted_seqs' column, since it's no longer needed
    final_unique_tert_contact_df = final_grouped.drop(columns=["sorted_seqs"])

    return final_unique_tert_contact_df


def update_unknown_motifs(
    potential_tert_contact_df: pd.DataFrame, motif_residue_dict: Dict
) -> pd.DataFrame:
    """
    Finds the "unknown" second motif in tertiary contacts.

    Args:
        potential_tert_contact_df (pd.DataFrame): DataFrame of potential tertiary contacts.
        motif_residue_dict (dict): dictionary listing which residues are in which motifs.

    Returns:
        potential_tert_contact_df (pd.DataFrame): DataFrame of potential tertiary contacts with unknowns found.

    """
    # Iterate through each row in the DataFrame
    for index, row in potential_tert_contact_df.iterrows():
        # Split motif names to compare the second element
        motif_1_split = row["motif_1"].split(".")
        motif_2_split = row["motif_2"].split(".")

        if row["motif_1"] == "unknown":
            residue = row["res_1"]
            for motif_name, residues in motif_residue_dict.items():
                motif_name_split = motif_name.split(".")
                # Check if residue is in residues and the second element matches
                if residue in residues and motif_2_split[1] == motif_name_split[1]:
                    potential_tert_contact_df.at[index, "motif_1"] = motif_name
                    break  # Once found, no need to continue searching

        if row["motif_2"] == "unknown":
            residue = row["res_2"]
            for motif_name, residues in motif_residue_dict.items():
                motif_name_split = motif_name.split(".")
                # Check if residue is in residues and the second element matches
                if residue in residues and motif_1_split[1] == motif_name_split[1]:
                    potential_tert_contact_df.at[index, "motif_2"] = motif_name
                    break  # Once found, no need to continue searching

    # Remove rows where either 'motif_1' or 'motif_2' still contains 'unknown'
    potential_tert_contact_df = potential_tert_contact_df[
        (potential_tert_contact_df["motif_1"] != "unknown")
        & (potential_tert_contact_df["motif_2"] != "unknown")
    ]

    return potential_tert_contact_df


def import_residues_csv(csv_dir: str) -> Dict:
    """
    Imports dictionary of residues in motif from CSV.

    Args:
        csv_dir (str): directory of where CSV outputs are stored.

    Returns:
        residues_dict (dict): dictionary of residues in motifs

    """
    csv_path = os.path.join(csv_dir, "residues_in_motif.csv")
    residues_dict = {}
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            motif_name = row["motif_name"]
            residues = row["residues"]
            residues_dict[motif_name] = residues.split(",")
    return residues_dict


def import_tert_contact_csv(csv_dir: str) -> pd.DataFrame:
    """
    Imports potential tertiary contact data from CSV.

    Args:
        csv_dir (str): Directory where CSV outputs are stored.

    Returns:
        interactions_from_csv (pd.DataFrame): Dataframe of the CSV data.

    """
    # Import the file as a DataFrame
    interactions_from_csv = pd.read_csv(
        os.path.join(csv_dir, "potential_tertiary_contacts.csv")
    )

    # Apply the function to each row and replace the values in type_1 and type_2
    interactions_from_csv["type_1"] = interactions_from_csv.apply(
        lambda row: assign_res_type(row["atom_1"], row["type_1"]), axis=1
    )
    interactions_from_csv["type_2"] = interactions_from_csv.apply(
        lambda row: assign_res_type(row["atom_2"], row["type_2"]), axis=1
    )

    return interactions_from_csv


def print_tert_contacts_to_cif(unique_tert_contact_df: pd.DataFrame) -> None:
    """
    Print tertiary contacts to CIF files.

    Args:
        unique_tert_contact_df (pd.DataFrame): DataFrame containing unique tertiary contacts.

    Returns:
        None
    """
    # Create directory for tertiary contacts if it doesn't exist
    os.makedirs("data/tertiary_contacts", exist_ok=True)
    print("Saving tertiary contacts to CIF files...")

    # Iterate through each row in the DataFrame
    for _, row in unique_tert_contact_df.iterrows():
        motif_1 = row["motif_1"]
        motif_2 = row["motif_2"]

        # Define the CIF file names
        motif_cif_1 = f"{motif_1}.cif"
        motif_cif_2 = f"{motif_2}.cif"

        # Find the paths to the CIF files
        directory_to_search = "data/motifs"
        path_to_cif_1 = find_cif_file(directory_to_search, motif_cif_1)
        path_to_cif_2 = find_cif_file(directory_to_search, motif_cif_2)

        # Define the output path for the merged CIF file
        motif_type_1 = motif_1.split(".")[0]
        motif_type_2 = motif_2.split(".")[0]
        tert_contact_name = f"{motif_1}.{motif_2}"
        sorted_motif_types = sorted([motif_type_1, motif_type_2])
        tert_contact_folder_name = f"{sorted_motif_types[0]}-{sorted_motif_types[1]}"
        tert_contact_out_path = (
            f"data/tertiary_contacts/{tert_contact_folder_name}/{tert_contact_name}.cif"
        )
        os.makedirs(f"data/tertiary_contacts/{tert_contact_folder_name}", exist_ok=True)
        # Print the tertiary contact name
        print(f"Processing: {tert_contact_name}")

        # Merge the CIF files
        try:
            merge_cif_files(
                file1_path=path_to_cif_1,
                file2_path=path_to_cif_2,
                output_path=tert_contact_out_path,
                lines_to_delete=24,  # Adjust this as needed
            )
        except TypeError as e:
            print(f"Failed to merge {motif_1} and {motif_2}: {e}")
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
