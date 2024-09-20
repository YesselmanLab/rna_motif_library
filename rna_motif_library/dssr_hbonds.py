import concurrent.futures
import math
import os
import numpy as np
import pandas as pd

from typing import List, Tuple

from pydssr.dssr_classes import DSSR_HBOND
from rna_motif_library.classes import DSSRRes, HBondInteraction, canon_amino_acid_list, SingleMotifInteraction, \
    PotentialTertiaryContact, Motif, HBondInteractionFactory, RNPInteraction
from rna_motif_library.settings import LIB_PATH


def save_interactions_to_disk(assembled_interaction_data: List[HBondInteraction], pdb: str) -> None:
    """
    Saves HBondInteraction objects to the disk.

    Args:
        assembled_interaction_data (list): list of HBondInteraction objects to write to disk
        pdb (str): name of PDB interaction is derived from

    Returns:
        None
    """

    for interaction in assembled_interaction_data:
        interaction_name = (
                str(pdb)
                + "."
                + interaction.res_1
                + "."
                + interaction.atom_1
                + "."
                + interaction.res_2
                + "."
                + interaction.atom_2
        )
        folder_path = os.path.join(
            LIB_PATH,
            "data/interactions",
            f"{DSSRRes(interaction.res_1).res_id}-{DSSRRes(interaction.res_2).res_id}",
        )
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{interaction_name}.cif")
        dataframe_to_cif(
            interaction.pdb, file_path=file_path, motif_name=interaction_name
        )


def build_complete_hbond_interaction(pre_assembled_interaction_data: List[HBondInteractionFactory],
                                     pdb_model_df: pd.DataFrame, pdb_name: str) -> List[HBondInteraction]:
    """
    Builds a complete HBondInteraction object from HBondInteractionFactory preliminary data

    Args:
        pre_assembled_interaction_data (list): pre-assembled HBondInteractionFactory data to draw data from
        pdb_model_df (pd.DataFrame): PDB dataframe
        pdb_name (str): name of source PDB

    Returns:
        built_interactions (list): a list of built HBondInteraction objects
    """

    built_interactions = []
    for interaction in pre_assembled_interaction_data:
        res_1 = DSSRRes(interaction.res_1)
        res_2 = DSSRRes(interaction.res_2)
        atom_1 = interaction.atom_1
        atom_2 = interaction.atom_2
        type_1 = interaction.residue_pair.split(":")[0]
        type_2 = interaction.residue_pair.split(":")[1]
        distance = interaction.distance
        pdb = get_interaction_pdb(res_1, res_2, pdb_model_df)
        first_atom, second_atom = extract_interacting_atoms(interaction, pdb)
        third_atom, fourth_atom = find_closest_atom(first_atom, pdb), find_closest_atom(
            second_atom, pdb
        )
        if first_atom.empty or second_atom.empty:
            # print(pdb)
            # print("EMPTY ATOM")
            continue
        # filter out protein-protein interactions
        if type_1 == "aa" and type_2 == "aa":
            continue
        dihedral_angle = calculate_bond_angle(
            first_atom, second_atom, third_atom, fourth_atom
        )

        built_interaction = HBondInteraction(
            interaction.res_1,
            interaction.res_2,
            atom_1,
            atom_2,
            type_1,
            type_2,
            distance,
            dihedral_angle,
            pdb,
            first_atom,
            second_atom,
            third_atom,
            fourth_atom,
            pdb_name,
        )
        built_interactions.append(built_interaction)

    return built_interactions


def get_interaction_pdb(res_1: DSSRRes, res_2: DSSRRes, pdb_model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtains PDB of combined interaction (not individual residues)

    Args:
        res_1 (DSSRRes): residue 1 in interaction obtained from DSSR/SNAP
        res_2 (DSSRRes): residue 2 in interaction obtained from DSSR/SNAP
        pdb_model_df (pd.DataFrame): dataframe with source PDB data

    Returns:
        res_1_res_2_result_df (pd.DataFrame): dataframe containing the overall interaction

    """
    res_1_chain_id, res_1_atom_type, res_1_res_id = (
        res_1.chain_id,
        res_1.res_id,
        res_1.num,
    )
    res_2_chain_id, res_2_atom_type, res_2_res_id = (
        res_2.chain_id,
        res_2.res_id,
        res_2.num,
    )
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

    return res_1_res_2_result_df


def assemble_interaction_data(unique_interaction_data: List[Tuple[str, str, str, str, str, str, str]]) -> \
List[HBondInteractionFactory]:
    """
    Loads data into intermediate HBondInteractionFactory from DSSR/SNAP output data.

    Args:
        unique_interaction_data (list): raw interaction data from DSSR/SNAP

    Returns:
        assembled_data (list): raw data loaded into the intermediate HBondInteractionFactory class for further processing

    """
    assembled_data = []
    # Load all data into HBondInteractionFactory class to prepare for processing
    for interaction in unique_interaction_data:
        # Filter out bad H-bonds and aa:aa
        if interaction[6] not in ["questionable", "unknown"] or interaction[5] not in [
            "aa:aa"
        ]:

            new_interaction_atom_1 = interaction[2]
            new_interaction_atom_2 = interaction[3]

            # Also process the weird ones with . in their name
            if "." in new_interaction_atom_1:
                new_interaction_atom_1 = interaction[2].split(".")[0]
            elif "." in new_interaction_atom_2:
                new_interaction_atom_2 = interaction[3].split(".")[0]
            hbond_interaction_assembly = HBondInteractionFactory(
                interaction[0],
                interaction[1],
                new_interaction_atom_1,
                new_interaction_atom_2,
                float(interaction[4]),
                interaction[5],
                interaction[6],
            )
            assembled_data.append(hbond_interaction_assembly)
    return assembled_data

def merge_hbond_interaction_data(rnp_interactions: List[RNPInteraction], hbonds: List[DSSR_HBOND]) -> List[
    Tuple[str, str, str, str, str, str, str]]:
    """
    Merges H-bond interaction data from DSSR and SNAP into one common data set.

    Args:
        rnp_interactions (list): list of RNPInteraction objects from SNAP
        hbonds (list): list of DSSR_HBOND objects from DSSR

    Returns:
        unique_interaction_data (list): list of tuples with interaction data

    """

    rnp_data = [
        (
            interaction.nt_atom.split("@")[1],
            interaction.aa_atom.split("@")[1],
            interaction.nt_atom.split("@")[0],
            interaction.aa_atom.split("@")[0],
            str(interaction.dist),
            interaction.type,
            "standard",
        )
        for interaction in rnp_interactions
    ]
    interaction_data = [
        (
            hbond.atom1_id.split("@")[1],
            hbond.atom2_id.split("@")[1],
            hbond.atom1_id.split("@")[0],
            hbond.atom2_id.split("@")[0],
            str(hbond.distance),
            hbond.residue_pair,
            hbond.donAcc_type,
        )
        for hbond in hbonds
    ]
    interaction_data.extend(rnp_data)
    unique_interaction_data = list(set(interaction_data))

    return unique_interaction_data


def extract_interacting_atoms(interaction: HBondInteractionFactory, pdb: pd.DataFrame):
    """
    Extracts interacting atoms from PDB data given interaction data from DSSR/SNAP.

    Args:
        interaction (HBondInteraction): HBondInteraction object
        pdb (pd.DataFrame): dataframe containing PDB structure information

    Returns:
        first_atom: PDB of first atom in interaction
        second_atom: PDB of second atom in interaction
    """
    atom_1 = interaction.atom_1
    atom_2 = interaction.atom_2

    res_1 = DSSRRes(interaction.res_1).res_id
    res_2 = DSSRRes(interaction.res_2).res_id

    chain_id_1 = DSSRRes(interaction.res_1).chain_id
    chain_id_2 = DSSRRes(interaction.res_2).chain_id

    res_id_1 = DSSRRes(interaction.res_1).num
    res_id_2 = DSSRRes(interaction.res_2).num

    first_atom = pdb[
        (pdb["auth_atom_id"] == atom_1)
        & (pdb["auth_comp_id"] == res_1)
        & (pdb["auth_asym_id"] == chain_id_1)
        & (pdb["auth_seq_id"] == res_id_1)
        ]
    second_atom = pdb[
        (pdb["auth_atom_id"] == atom_2)
        & (pdb["auth_comp_id"] == res_2)
        & (pdb["auth_asym_id"] == chain_id_2)
        & (pdb["auth_seq_id"] == res_id_2)
        ]

    if first_atom.empty:
        # Check for common prefixes or alternate namings
        prefixes = ["O1P", "O2P", "O3P", "OP1", "OP2", "OP3", "O2"]
        for prefix in prefixes:
            if prefix in atom_1:
                first_atom = pdb[
                    (
                            pdb["auth_atom_id"].str.contains(prefix.replace("P", ""))
                            & (pdb["auth_comp_id"] == res_1)
                            & (pdb["auth_asym_id"] == chain_id_1)
                            & (pdb["auth_seq_id"] == res_id_1)
                    )
                ]
                if not first_atom.empty:
                    break

    if second_atom.empty:
        # Check for common prefixes or alternate namings
        prefixes = ["O1P", "O2P", "O3P", "OP1", "OP2", "OP3", "O2"]
        for prefix in prefixes:
            if prefix in atom_2:
                second_atom = pdb[
                    (
                            pdb["auth_atom_id"].str.contains(prefix.replace("P", ""))
                            & (pdb["auth_comp_id"] == res_2)
                            & (pdb["auth_asym_id"] == chain_id_2)
                            & (pdb["auth_seq_id"] == res_id_2)
                    )
                ]
                if not first_atom.empty:
                    break

    return first_atom, second_atom


def print_residues_in_motif_to_csv(motifs_per_pdb: List[List[Motif]],
                                   csv_dir: str) -> None:
    """
    Prints all obtained motif/interaction data to a CSV.

    Args:
        motifs_per_pdb (list): List of all motifs in PDB.
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
