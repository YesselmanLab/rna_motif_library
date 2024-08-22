import math
import os
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, IO
from classes import (
    DSSRRes,
    HBondInteraction,
    extract_longest_numeric_sequence,
    canon_amino_acid_list, HBondInteractionFactory, SingleMotifInteraction, PotentialTertiaryContact, Motif,
)


def print_obtained_motif_interaction_data_to_csv(motifs_per_pdb: List[List[Motif]],
                                                 all_potential_tert_contacts: List[List[PotentialTertiaryContact]],
                                                 all_interactions: List[List[HBondInteraction]],
                                                 all_single_motif_interactions: List[List[SingleMotifInteraction]],
                                                 csv_dir: str) -> None:
    """
    Prints all obtained motif/interaction data to a CSV.

    Args:
        motifs_per_pdb: List of all motifs in PDB.
        all_potential_tert_contacts: List of all potential tertiary contacts.
        all_interactions: List of all h-bond interactions.
        all_single_motif_interactions: List of all single motif interactions.
        csv_dir (str): Directory where CSV outputs are printed to.

    Returns:
        None

    """

    # After you have motifs, print some data to a CSV
    # First thing to print would be a list of all motifs and the residues inside them
    # We can use this list when identifying tertiary contacts
    # residues_in_motif.csv
    residue_data = []
    for motifs in motifs_per_pdb:
        for motif in motifs:
            motif_name = motif.motif_name
            motif_residues = ",".join(motif.res_list)
            # Append the data as a dictionary to the list
            residue_data.append({"motif_name": motif_name, "residues": motif_residues})

    residues_in_motif_df = pd.DataFrame(residue_data)
    residues_in_motif_df.to_csv(
        os.path.join(csv_dir, "residues_in_motif.csv"), index=False
    )

    # print all potential tertiary contacts to CSV
    potential_tert_contact_data = []
    for potential_contacts in all_potential_tert_contacts:
        for potential_contact in potential_contacts:
            motif_1 = potential_contact.motif_1
            motif_2 = potential_contact.motif_2
            res_1 = potential_contact.res_1
            res_2 = potential_contact.res_2
            atom_1 = potential_contact.atom_1
            atom_2 = potential_contact.atom_2
            type_1 = potential_contact.type_1
            type_2 = potential_contact.type_2
            # Interactions with amino acids are absolutely not tertiary contacts
            if (
                    type_1 == "aa"
                    or type_2 == "aa"
                    or type_1 == "ligand"
                    or type_2 == "ligand"
            ):
                continue

            # Append the filtered data to the list as a dictionary
            potential_tert_contact_data.append(
                {
                    "motif_1": motif_1,
                    "motif_2": motif_2,
                    "res_1": res_1,
                    "res_2": res_2,
                    "atom_1": atom_1,
                    "atom_2": atom_2,
                    "type_1": type_1,
                    "type_2": type_2,
                }
            )
    # Create a DataFrame from the list of dictionaries and spit to CSV
    potential_tert_contact_df = pd.DataFrame(potential_tert_contact_data)
    potential_tert_contact_df.to_csv(
        os.path.join(csv_dir, "potential_tertiary_contacts.csv"), index=False
    )

    # Next we print a detailed list of every interaction we've found
    # interactions_detailed.csv
    interaction_data = []
    for interaction_set in all_interactions:
        for interaction in interaction_set:
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = interaction.distance
            angle = interaction.angle
            pdb_name = interaction.pdb_name
            mol_1 = DSSRRes(res_1).res_id
            mol_2 = DSSRRes(res_2).res_id
            # filter out ligands
            if type_1 == "ligand" or type_2 == "ligand":
                continue
            # Append the data to the list as a dictionary
            interaction_data.append(
                {
                    "pdb_name": pdb_name,
                    "res_1": res_1,
                    "res_2": res_2,
                    "mol_1": mol_1,
                    "mol_2": mol_2,
                    "atom_1": atom_1,
                    "atom_2": atom_2,
                    "type_1": type_1,
                    "type_2": type_2,
                    "distance": distance,
                    "angle": angle,
                }
            )
    # Create a DataFrame from the list of dictionaries
    interactions_detailed_df = pd.DataFrame(interaction_data)
    interactions_detailed_df.to_csv(
        os.path.join(csv_dir, "interactions_detailed.csv"), index=False
    )

    # Single motif interactions
    # single_motif_interactions.csv
    single_motif_interaction_data = []
    for interaction_set in all_single_motif_interactions:
        for interaction in interaction_set:
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = interaction.distance
            angle = interaction.angle
            motif_name = interaction.motif_name
            mol_1 = DSSRRes(res_1).res_id
            mol_2 = DSSRRes(res_2).res_id
            # filter out ligands
            if type_1 == "ligand" or type_2 == "ligand":
                continue
            # Append the data to the list as a dictionary
            single_motif_interaction_data.append(
                {
                    "motif_name": motif_name,
                    "res_1": res_1,
                    "res_2": res_2,
                    "mol_1": mol_1,
                    "mol_2": mol_2,
                    "atom_1": atom_1,
                    "atom_2": atom_2,
                    "type_1": type_1,
                    "type_2": type_2,
                    "distance": distance,
                    "angle": angle,
                }
            )
    single_motif_interaction_data_df = pd.DataFrame(single_motif_interaction_data)
    single_motif_interaction_data_df.to_csv(
        os.path.join(csv_dir, "single_motif_interaction.csv"), index=False
    )


def find_closest_atom(
        atom_A: pd.DataFrame, whole_interaction: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds the closest atom to a given atom within a set of interactions based on Euclidean distance.
    This function iterates through each atom in `whole_interaction`, calculates the distance to `atom_A`,
    and keeps track of the atom with the minimum distance. Returns a DataFrame containing the closest atom.

    Args:
        atom_A (pd.DataFrame): A DataFrame representing the atom to which distance is measured.
        whole_interaction (pd.DataFrame): A DataFrame containing multiple atoms with which the distance will be compared.

    Returns:
        min_distance_row_df (pd.DataFrame): A DataFrame representing the atom closest to `atom_A` from `whole_interaction`.

    """
    min_distance_row = None
    min_distance = float("inf")
    for _, row in whole_interaction.iterrows():
        row_df = row.to_frame().T
        current_distance = calc_distance(atom_A, row_df)
        if 0 < current_distance < min_distance:
            min_distance = current_distance
            min_distance_row = row
    min_distance_row_df = (
        min_distance_row.to_frame().T
        if min_distance_row is not None
        else pd.DataFrame()
    )
    return min_distance_row_df


def calc_distance(atom_df1: pd.DataFrame, atom_df2: pd.DataFrame) -> float:
    """
    Calculates the Euclidean distance between two atoms using their Cartesian coordinates.
    The function tries to extract the 'Cartn_x', 'Cartn_y', and 'Cartn_z' coordinates from each DataFrame.
    If the coordinates are not accessible or an error occurs, it returns 0.

    Args:
        atom_df1 (pd.DataFrame): DataFrame containing the coordinates of the first atom.
        atom_df2 (pd.DataFrame): DataFrame containing the coordinates of the second atom.

    Returns:
        distance (float): The Euclidean distance between two points defined by the coordinates in atom_df1 and atom_df2.

    Raises:
        IndexError: Returns 0 if there is an error accessing coordinates.

    """
    try:
        x1 = atom_df1["Cartn_x"].tolist()[0]
        y1 = atom_df1["Cartn_y"].tolist()[0]
        z1 = atom_df1["Cartn_z"].tolist()[0]
        x2 = atom_df2["Cartn_x"].tolist()[0]
        y2 = atom_df2["Cartn_y"].tolist()[0]
        z2 = atom_df2["Cartn_z"].tolist()[0]
        distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
    except IndexError:
        distance = 0
    return distance


def calculate_bond_angle(
        center_atom: pd.DataFrame,
        second_atom: pd.DataFrame,
        carbon_atom: pd.DataFrame,
        fourth_atom: pd.DataFrame,
) -> float:
    """
    Calculates the bond angle and returns the angle with the atoms used in the calculation.

    Args:
        center_atom (pd.DataFrame): DataFrame containing the Cartesian coordinates for the center atom.
        second_atom (pd.DataFrame): DataFrame containing the Cartesian coordinates for the second atom.
        carbon_atom (pd.DataFrame): DataFrame containing the Cartesian coordinates for the carbon atom.
        fourth_atom (pd.DataFrame): DataFrame containing the Cartesian coordinates for the fourth atom.

    Returns:
        angle_deg (str): The calculated bond angle in degrees as a string.
        center_atom_data (tuple): A tuple with the center atom type and its coordinates.
        second_atom_data (tuple): A tuple with the second atom type and its coordinates.
        carbon_atom_data (tuple): A tuple with the carbon atom type and its coordinates.

    """

    # Extract coordinates
    x1, y1, z1 = (
        center_atom["Cartn_x"].to_list()[0],
        center_atom["Cartn_y"].to_list()[0],
        center_atom["Cartn_z"].to_list()[0],
    )
    x2, y2, z2 = (
        second_atom["Cartn_x"].to_list()[0],
        second_atom["Cartn_y"].to_list()[0],
        second_atom["Cartn_z"].to_list()[0],
    )
    x3, y3, z3 = (
        carbon_atom["Cartn_x"].to_list()[0],
        carbon_atom["Cartn_y"].to_list()[0],
        carbon_atom["Cartn_z"].to_list()[0],
    )
    x4, y4, z4 = (
        fourth_atom["Cartn_x"].to_list()[0],
        fourth_atom["Cartn_y"].to_list()[0],
        fourth_atom["Cartn_z"].to_list()[0],
    )
    # Tuples passed as points
    P = (x1, y1, z1)
    A = (x2, y2, z2)
    C = (x3, y3, z3)
    F = (x4, y4, z4)
    # Calculate vectors
    vector_AP = np.array(A) - np.array(P)
    vector_PC = np.array(C) - np.array(P)
    vector_FA = np.array(F) - np.array(A)
    # Get cross products for planes
    vector_n1 = np.cross(vector_AP, vector_PC)
    vector_n2 = np.cross(vector_AP, vector_FA)
    dot_product = np.dot(vector_n1, vector_n2)
    magnitude_n1 = np.linalg.norm(vector_n1)
    magnitude_n2 = np.linalg.norm(vector_n2)
    cos_theta = dot_product / (magnitude_n1 * magnitude_n2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = float(np.degrees(angle_rad))
    return angle_deg


def assign_res_type(name: str, res_type: str) -> str:
    """
    Assign base, phosphate, sugar, or amino acid given residue (nt/aa) and atom.

    Args:
        name (str): The name of the residue.
        res_type (str): The type of the residue (e.g., "aa" for amino acid).

    Returns:
        string (str): The assigned residue type as a string.

    """
    if res_type == "aa":
        return "aa"
    else:
        if name in canon_amino_acid_list:
            return "aa"
        elif "P" in name:
            return "phos"
        elif name.endswith("'"):
            return "sugar"
        else:
            return "base"


def dataframe_to_cif(df: pd.DataFrame, file_path: str, motif_name: str) -> None:
    """
    Converts a DataFrame to CIF format and writes it to a file.
    Custom-built function for our purposes.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        file_path (str): The path to the output CIF file.
        motif_name (str): The name of the PDB to be not printed.

    Returns:
        None

    """
    with open(file_path, "w") as f:
        # Write the CIF header section; len(row) = 21
        f.write("data_\n")
        f.write("_entry.id " + motif_name + "\n")
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")  # 0
        f.write("_atom_site.id\n")  # 1
        f.write("_atom_site.type_symbol\n")  # 2
        f.write("_atom_site.label_atom_id\n")  # 3
        f.write("_atom_site.label_alt_id\n")  # 4
        f.write("_atom_site.label_comp_id\n")  # 5
        f.write("_atom_site.label_asym_id\n")  # 6
        f.write("_atom_site.label_entity_id\n")  # 7
        f.write("_atom_site.label_seq_id\n")  # 8
        f.write("_atom_site.pdbx_PDB_ins_code\n")  # 9
        f.write("_atom_site.Cartn_x\n")  # 10
        f.write("_atom_site.Cartn_y\n")  # 11
        f.write("_atom_site.Cartn_z\n")  # 12
        f.write("_atom_site.occupancy\n")  # 13
        f.write("_atom_site.B_iso_or_equiv\n")  # 14
        f.write("_atom_site.pdbx_formal_charge\n")  # 15
        f.write("_atom_site.auth_seq_id\n")  # 16
        f.write("_atom_site.auth_comp_id\n")  # 17
        f.write("_atom_site.auth_asym_id\n")  # 18
        f.write("_atom_site.auth_atom_id\n")  # 19
        f.write("_atom_site.pdbx_PDB_model_num\n")  # 20
        # Write the data from the DataFrame (formatting)
        for row in df.itertuples(index=False):
            f.write(
                "{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}{:<10}{:<10}"
                "{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}\n".format(
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                    str(row[7]),
                    str(row[8]),
                    str(row[9]),
                    str(row[10]),
                    str(row[11]),
                    str(row[12]),
                    str(row[13]),
                    str(row[14]),
                    str(row[15]),
                    str(row[16]),
                    str(row[17]),
                    str(row[18]),
                    str(row[19]),
                    str(row[20]),
                )
            )
