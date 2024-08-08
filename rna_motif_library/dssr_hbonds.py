import math
import os
from typing import Tuple, List, Dict, IO, Any

import numpy as np
import pandas as pd

canon_amino_acid_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


class HBondInteraction:
    def __init__(self, res_1, res_2, atom_1, atom_2, type_1, type_2, distance, angle, pdb, first_atom_df,
                 second_atom_df, third_atom_df, fourth_atom_df):
        """
        Holds data for H-bond interaction

        Args:
            res_1 (str): residue 1 in the interaction
            res_2 (str): residue 2 in the interaction
            atom_1 (str): atom 1 in the interaction
            atom_2 (str): atom 2 in the interaction
            type_1 (str): residue type 1 in the interaction
            type_2 (str): residue type 2 in the interaction
            distance (float): distance between atoms in interaction
            angle (float): dihedral angle between two residues
            pdb (pd.DataFrame): interaction PDB

            first_atom_df (pd.DataFrame): PDB of the first atom in the interaction
            second_atom_df (pd.DataFrame): PDB of the second atom in the interaction
            third_atom_df (pd.DataFrame): PDB of the third atom connected to the first atom
            fourth_atom_df (pd.DataFrame): PDB of the fourth atom connected to the second atom

        """
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle
        self.pdb = pdb
        self.first_atom_df = first_atom_df
        self.second_atom_df = second_atom_df
        self.third_atom_df = third_atom_df
        self.fourth_atom_df = fourth_atom_df


class HBondInteractionFactory:
    """
    Intermediate class to assist in building complete HBondInteraction data

    Args:
        res_1 (str): residue 1 ID
        res_2 (str): residue 2 ID
        atom_1 (str): atom 1 ID
        atom_2 (str): atom 2 ID
        distance (float): distance between interacting atoms (in angstroms)
        residue_pair (str): pair comparing the types as returned by DSSR
        quality (str): quality of h-bond

    """

    def __init__(self, res_1, res_2, atom_1, atom_2, distance, residue_pair, quality):
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.distance = distance
        self.residue_pair = residue_pair
        self.quality = quality


def extract_hbonds_from_dssr():
    # for writing to interactions.csv
    start_interactions_dict = {
        "base:base": 0,
        "base:sugar": 0,
        "base:phos": 0,
        "sugar:base": 0,
        "sugar:sugar": 0,
        "sugar:phos": 0,
        "phos:base": 0,
        "phos:sugar": 0,
        "phos:phos": 0,
        "base:aa": 0,
        "sugar:aa": 0,
        "phos:aa": 0,
    }


def load_interactions_as_classes(unique_inters):
    unique_inters_loaded_as_classes = []
    for inter in unique_inters:
        res_1 = inter[0]
        res_2 = inter[1]
        atom_1 = inter[2]
        atom_2 = inter[3]
        distance = inter[4]
        res_part_1 = inter[5]
        res_part_2 = inter[6]
        unique_hbond_interaction = HBondInteraction(res_1, res_2, atom_1, atom_2, distance, res_part_1, res_part_2)
        unique_inters_loaded_as_classes.append(unique_hbond_interaction)

    return unique_inters_loaded_as_classes


def print_interactions_to_csv(
        alpha_sorted_types: List[str],
        name_inter: str,
        res_1_res_2_result_df: pd.DataFrame,
        res_1_type: str,
        res_2_type: str,
        atom_1: str,
        atom_2: str,
        start_interactions_dict: Dict[str, int],
        csv_file: IO[str],
        motif_name: str,
        res_1: str,
        res_2: str,
        distance_ext: str,
        bond_angle_degrees: str,
) -> None:
    """
    Print interaction details to a CSV file and save the data to a CIF file.

    Args:
        alpha_sorted_types (list): Alphabetically sorted types of residues.
        name_inter (str): Name of the interaction.
        res_1_res_2_result_df (pd.DataFrame): DataFrame containing the result of the interaction.
        res_1_type (str): Type of the first residue.
        res_2_type (str): Type of the second residue.
        atom_1 (str): Atom identifier for the first residue.
        atom_2 (str): Atom identifier for the second residue.
        start_interactions_dict (dict): Dictionary to track interaction counts.
        csv_file (CSV file): CSV file to write the interaction data.
        motif_name (str): Name of the motif.
        res_1 (str): Identifier for the first residue.
        res_2 (str): Identifier for the second residue.
        distance_ext (str): Distance between the residues.
        bond_angle_degrees (str): Bond angle between the residues.

    Returns:
        None

    """
    # folder assignment
    folder_name = alpha_sorted_types[0] + "-" + alpha_sorted_types[1]
    ind_folder_path = "data/interactions/" + folder_name + "/"
    os.makedirs(ind_folder_path, exist_ok=True)
    name_inter_2 = name_inter.replace("/", "-")
    ind_inter_path = ind_folder_path + name_inter_2
    res_1_res_2_result_df.fillna(0, inplace=True)
    dataframe_to_cif(
        res_1_res_2_result_df, file_path=f"{ind_inter_path}.cif", motif_name=name_inter
    )
    if len(res_1_type) == 1:
        nt_1 = "nt"
    else:
        nt_1 = "aa"

    if len(res_2_type) == 1:
        nt_2 = "nt"
    else:
        nt_2 = "aa"
    type_1 = assign_res_type(atom_1, nt_1)
    type_2 = assign_res_type(atom_2, nt_2)
    # Here we count the stuff for interactions.csv
    # first set the class
    if type_1 != "aa":
        hbond_class = type_1 + ":" + type_2
    else:
        hbond_class = type_2 + ":" + type_1
    # then increment the dictionary
    start_interactions_dict[hbond_class] += 1
    csv_file.write(
        motif_name
        + ","
        + res_1
        + ","
        + res_2
        + ","
        + res_1_type
        + ","
        + res_2_type
        + ","
        + atom_1
        + ","
        + atom_2
        + ","
        + distance_ext
        + ","
        + bond_angle_degrees
        + ","
        + nt_1
        + ","
        + nt_2
        + ","
        + type_1
        + ","
        + type_2
        + "\n"
    )



def find_atoms(residue: pd.DataFrame, atom_id: str) -> pd.DataFrame:
    """
    Finds atoms with alternate namings.

    Args:
        residue (pd.DataFrame): residue to be processed
        atom_id (str): ID of atom to be processed

    Returns:
        atom (pd.DataFrame): phosphate atom in residue

    """
    atom = residue[residue["auth_atom_id"] == atom_id]

    if atom.empty:
        # Check for common prefixes or alternate namings
        prefixes = ["O1P", "O2P", "O3P", "OP1", "OP2", "OP3", "O2"]
        for prefix in prefixes:
            if prefix in atom_id:
                atom = residue[
                    residue["auth_atom_id"].str.contains(prefix.replace("P", ""))
                ]
                if not atom.empty:
                    break
    return atom


def extract_residues_from_interaction_source(
        interaction: List[str], pdb_model_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, str, str, str, str]:
    """
    Extract residues from interaction source.

    Args:
        interaction (list): Interaction details containing residue and atom information.
        pdb_model_df (pd.DataFrame): DataFrame containing PDB model data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, str, str, str, str]:
            - Combined DataFrame of both residues' interactions.
            - DataFrame of the first residue's interactions.
            - DataFrame of the second residue's interactions.
            - Identifier of the first residue.
            - Identifier of the second residue.
            - Type of the first residue.
            - Type of the second residue.
            - Atom identifier of the first residue.
            - Atom identifier of the second residue.
            - Distance between the interacting residues.

    """
    res_1 = interaction[0]
    res_2 = interaction[1]
    res_1_chain_id, res_1_res_data = res_1.split(".")
    res_2_chain_id, res_2_res_data = res_2.split(".")
    if "/" in res_1:
        res_1_split = res_1.split(".")
        res_1_inside = res_1_split[1]
        res_1_res_data_spl = res_1_inside.split("/")
        res_1_res_data = res_1_res_data_spl[1]
    if "/" in res_2:
        res_2_split = res_2.split(".")
        res_2_inside = res_2_split[1]
        res_2_res_data_spl = res_2_inside.split("/")
        res_2_res_data = res_2_res_data_spl[1]
    res_1_res_id = extract_longest_numeric_sequence(res_1_res_data)
    res_2_res_id = extract_longest_numeric_sequence(res_2_res_data)
    res_1_inter_chain = pdb_model_df[
        pdb_model_df["auth_asym_id"].astype(str) == str(res_1_chain_id)
        ]
    res_2_inter_chain = pdb_model_df[
        pdb_model_df["auth_asym_id"].astype(str) == str(res_2_chain_id)
        ]
    res_1_inter_res = res_1_inter_chain[
        res_1_inter_chain["auth_seq_id"].astype(str) == str(res_1_res_id)
        ]
    res_2_inter_res = res_2_inter_chain[
        res_2_inter_chain["auth_seq_id"].astype(str) == str(res_2_res_id)
        ]
    res_1_res_2_result_df = pd.concat(
        [res_1_inter_res, res_2_inter_res], axis=0, ignore_index=True
    )
    # write interactions to CSV
    res_1_type_list = res_1_inter_res["auth_comp_id"].unique().tolist()
    res_2_type_list = res_2_inter_res["auth_comp_id"].unique().tolist()
    res_1_type = res_1_type_list[0]
    res_2_type = res_2_type_list[0]
    atom_1 = interaction[2]
    atom_2 = interaction[3]
    distance_ext = interaction[4]
    return (
        res_1_res_2_result_df,
        res_1_inter_res,
        res_2_inter_res,
        res_1,
        res_2,
        res_1_type,
        res_2_type,
        atom_1,
        atom_2,
        distance_ext,
    )


def extract_longest_numeric_sequence(input_string: str) -> str:
    """
    Extracts the longest numeric sequence from a given string.

    Args:
        input_string (str): The string to extract the numeric sequence from.

    Returns:
        longest_sequence (str): The longest numeric sequence found in the input string.

    """
    longest_sequence = ""
    current_sequence = ""
    for c in input_string:
        if c.isdigit() or (
                c == "-" and (not current_sequence or current_sequence[0] == "-")
        ):
            current_sequence += c
            if len(current_sequence) >= len(longest_sequence):
                longest_sequence = current_sequence
        else:
            current_sequence = ""
    return longest_sequence


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
) -> str:
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
    angle_deg = str(np.degrees(angle_rad))
    # Label and return the atoms used in calculation
    center_atom_type = center_atom["auth_atom_id"].to_list()[0]
    carbon_atom_type = carbon_atom["auth_atom_id"].to_list()[0]
    second_atom_type = second_atom["auth_atom_id"].to_list()[0]
    center_atom_data = (center_atom_type, x1, y1, z1)
    second_atom_data = (second_atom_type, x2, y2, z2)
    carbon_atom_data = (carbon_atom_type, x3, y3, z3)
    return angle_deg


def assign_res_type(name: str, res_type: str) -> str:
    """
    Assign base, phosphate, sugar, or amino acid in interactions_detailed.csv.

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

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        file_path (str): The path to the output CIF file.
        motif_name (str): The name of the motif.

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
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11],
                    row[12],
                    row[13],
                    row[14],
                    row[15],
                    row[16],
                    row[17],
                    row[18],
                    row[19],
                    row[20],
                )
            )
