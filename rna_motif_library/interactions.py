import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

from pydssr.dssr_classes import DSSR_HBOND, DSSR_PAIR

from rna_motif_library.classes import (
    X3DNAInteraction,
    Hbond,
    X3DNAResidue,
    X3DNAResidueFactory,
    X3DNAPair,
    canon_amino_acid_list,
    sanitize_x3dna_atom_name,
)
from rna_motif_library.logger import get_logger
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.settings import LIB_PATH

log = get_logger("interactions")


def dataframe_to_cif(df: pd.DataFrame, file_path: str) -> None:
    with open(file_path, "w") as f:
        # Write the CIF header section
        f.write("data_\n")
        f.write("_entry.id " + file_path.split(".")[0] + "\n")
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.auth_atom_id\n")
        f.write("_atom_site.auth_comp_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")

        # Write the data from the DataFrame
        for _, row in df.iterrows():
            f.write(
                "{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}\n".format(
                    str(row["group_PDB"]),
                    str(row["id"]),
                    str(row["auth_atom_id"]),
                    str(row["auth_comp_id"]),
                    str(row["auth_asym_id"]),
                    str(row["auth_seq_id"]),
                    str(row["Cartn_x"]),
                    str(row["Cartn_y"]),
                    str(row["Cartn_z"]),
                )
            )


# handles converting X3DNA to current objects ##########################################


def get_x3dna_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_x3dna_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
    return all_interactions
    # df_atoms = pd.read_parquet(pdb_df_path)
    # hbond_infos = build_complete_hbond_interaction(all_interactions, pdb_name, df_atoms)
    # return all_interactions


def convert_x3dna_hbonds_to_interactions(
    hbonds: List[DSSR_HBOND],
) -> List[X3DNAInteraction]:
    # Determine type based on atom name
    def get_atom_type(atom):
        if atom.startswith(("P", "OP")):
            return "phos"
        elif atom.startswith(
            ("O2'", "O3'", "O4'", "O5'", "C1'", "C2'", "C3'", "C4'", "C5'")
        ):
            return "sugar"
        else:
            return "base"

    rna_interactions = []
    for hbond in hbonds:
        atom_1, res_1 = hbond.atom1_id.split("@")
        atom_2, res_2 = hbond.atom2_id.split("@")
        atom_1 = sanitize_x3dna_atom_name(atom_1)
        atom_2 = sanitize_x3dna_atom_name(atom_2)
        type_1 = get_atom_type(atom_1)
        type_2 = get_atom_type(atom_2)
        for aa in canon_amino_acid_list:
            if aa in res_1:
                type_1 = "aa"
            if aa in res_2:
                type_2 = "aa"

        if type_1 == "aa" and type_2 == "aa":
            continue
        # Swap if res_1 is larger than res_2
        if res_1 > res_2:
            atom_1, atom_2 = atom_2, atom_1
            res_1, res_2 = res_2, res_1
            type_1, type_2 = type_2, type_1
        res_1 = X3DNAResidueFactory.create_from_string(res_1)
        res_2 = X3DNAResidueFactory.create_from_string(res_2)
        rna_interactions.append(
            X3DNAInteraction(
                atom_1, res_1, atom_2, res_2, float(hbond.distance), type_1, type_2
            )
        )
    return rna_interactions


def merge_hbond_interaction_data(
    rna_interactions: List[X3DNAInteraction], rnp_interactions: List[X3DNAInteraction]
) -> List[X3DNAInteraction]:
    """
    Merges RNA-RNA and RNA-protein interaction data and removes duplicates and protein-protein interactions.

    Args:
        rna_interactions (list): List of X3DNAInteraction objects from RNA-RNA interactions
        rnp_interactions (list): List of X3DNAInteraction objects from RNA-protein interactions

    Returns:
        unique_interactions (list): List of unique X3DNAInteraction objects, excluding protein-protein interactions
    """
    # Combine all interactions
    all_interactions = rna_interactions + rnp_interactions

    # Create a list of interactions, excluding protein-protein interactions
    unique_interactions = set()
    for interaction in all_interactions:
        # Skip if distance is greater than 3.3 this is not a real hbond
        if interaction.distance > 3.3:
            continue
        # Only keep interactions that aren't between two proteins
        is_protein_protein = (
            interaction.atom_type_1 == "aa" and interaction.atom_type_2 == "aa"
        )
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)


def parse_hbond_description(hbonds_desc: str) -> List[Tuple[str, str, float]]:
    """Parse DSSR hbond description into list of (atom1, atom2, distance) tuples"""
    hbond_details = []
    for hbond in hbonds_desc.split(","):
        # Extract the two atoms and distance
        if hbond.find("-") != -1:
            atoms = hbond.split("-")
        else:
            atoms = hbond.split("*")
        if len(atoms) != 2:
            continue
        # Get first atom, removing any parenthetical descriptions
        atom1 = atoms[0].split("(")[0].strip()
        # Get second atom and distance
        atom2_dist = atoms[1].split("[")
        if len(atom2_dist) != 2:
            continue
        atom2 = atom2_dist[0].split("(")[0].strip()
        distance = float(atom2_dist[1].strip("]"))
        # Skip if distance is greater than 3.3 this is not a real hbond
        if distance > 3.3:
            continue
        atom1 = sanitize_x3dna_atom_name(atom1)
        atom2 = sanitize_x3dna_atom_name(atom2)
        hbond_details.append((atom1, atom2, distance))
    return hbond_details


def get_x3dna_pairs(pairs: Dict[str, DSSR_PAIR], interactions: List[X3DNAInteraction]):
    """Get X3DNA pairs from DSSR pairs"""
    x3dna_pairs = []
    for pair in pairs.values():
        res_1 = X3DNAResidueFactory.create_from_string(pair.nt1.nt_id)
        res_2 = X3DNAResidueFactory.create_from_string(pair.nt2.nt_id)
        # Parse hbond description into atom pairs and distances
        hbond_details = parse_hbond_description(pair.hbonds_desc)
        if len(hbond_details) == 0:
            continue
        pair_interactions = []
        for interaction in interactions:
            if interaction.res_1 == res_1 and interaction.res_2 == res_2:
                pair_interactions.append(interaction)
            elif interaction.res_1 == res_2 and interaction.res_2 == res_1:
                pair_interactions.append(interaction)
        if len(pair_interactions) == 0:
            continue
        x3dna_pairs.append(
            X3DNAPair(res_1, res_2, pair_interactions, pair.name, pair.LW)
        )
    return x3dna_pairs


# handles converting to final objects ##################################################


def get_closest_atoms():
    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]
    return closest_atoms


def get_atom_coords(
    atom_name: str, residue: X3DNAResidue, df_atoms: pd.DataFrame
) -> Tuple[float, float, float]:
    df_atom = df_atoms[
        (df_atoms["auth_atom_id"] == atom_name)
        & (df_atoms["auth_comp_id"] == residue.res_id)
        & (df_atoms["auth_asym_id"] == residue.chain_id)
        & (df_atoms["auth_seq_id"] == residue.num)
    ]
    if df_atom.empty:
        return None
    return df_atom[["Cartn_x", "Cartn_y", "Cartn_z"]].values[0]


def find_closest_atom(
    atom_name: str,
    residue: X3DNAResidue,
    df_atoms: pd.DataFrame,
    closest_atoms: Dict[str, str],
) -> str:
    """
    Finds the closest atom to the given atom in a residue, either using a predefined dictionary
    or by calculating distances to all atoms in the residue.

    Args:
        atom_name (str): Name of the atom to find closest to
        residue (X3DNAResidue): Residue containing the atoms
        df_atoms (pd.DataFrame): DataFrame with atom coordinates
        closest_atoms (Dict[str, str]): Dictionary mapping residue-atom pairs to closest atoms

    Returns:
        str: Name of the closest atom found
    """
    # First try looking up in dictionary
    key = f"{residue.res_id}-{atom_name}"
    if key in closest_atoms:
        return closest_atoms[key]

    # If not in dictionary, calculate distances to all atoms in residue
    target_coords = get_atom_coords(atom_name, residue, df_atoms)
    if target_coords is None:
        return None

    # Get all atoms in the residue
    residue_atoms = df_atoms[
        (df_atoms["auth_comp_id"] == residue.res_id)
        & (df_atoms["auth_asym_id"] == residue.chain_id)
        & (df_atoms["auth_seq_id"] == residue.num)
    ]

    min_dist = float("inf")
    closest_atom = None

    # Calculate distance to each atom
    for _, atom in residue_atoms.iterrows():
        if atom["auth_atom_id"] == atom_name:
            continue

        atom_coords = np.array([atom["Cartn_x"], atom["Cartn_y"], atom["Cartn_z"]])
        dist = np.linalg.norm(target_coords - atom_coords)

        if dist < min_dist:
            min_dist = dist
            closest_atom = atom["auth_atom_id"]

    return closest_atom


def calculate_dihedral_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """
    Calculate the dihedral angle between 4 points in 3D space.

    Args:
        p1 (np.ndarray): Coordinates of first point
        p2 (np.ndarray): Coordinates of second point
        p3 (np.ndarray): Coordinates of third point
        p4 (np.ndarray): Coordinates of fourth point

    Returns:
        float: Dihedral angle in degrees
    """
    # Calculate vectors between points
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    # Calculate normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # Normalize normal vectors
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    # Calculate angle between normal vectors
    cos_angle = np.dot(n1, n2)
    # Handle numerical errors
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = np.arccos(cos_angle)
    # Determine sign of angle
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return round(np.degrees(angle), 1)


def get_hbonds(
    pdb_name: str,
    df_atoms: pd.DataFrame,
    x3dna_interactions: List[X3DNAInteraction],
    x3dna_pairs: List[X3DNAPair],
) -> List[Hbond]:
    hbonds = []
    pair_dict = {}
    for pair in x3dna_pairs:
        pair_dict[pair.res_1.get_str() + "-" + pair.res_2.get_str()] = pair
        pair_dict[pair.res_2.get_str() + "-" + pair.res_1.get_str()] = pair
    closest_atoms = get_closest_atoms()
    for interaction in x3dna_interactions:
        closest_atom_1 = find_closest_atom(
            interaction.atom_1, interaction.res_1, df_atoms, closest_atoms
        )
        closest_atom_2 = find_closest_atom(
            interaction.atom_2, interaction.res_2, df_atoms, closest_atoms
        )
        closest_atom_1_coords = get_atom_coords(
            closest_atom_1, interaction.res_1, df_atoms
        )
        closest_atom_2_coords = get_atom_coords(
            closest_atom_2, interaction.res_2, df_atoms
        )
        atom_1_coords = get_atom_coords(interaction.atom_1, interaction.res_1, df_atoms)
        atom_2_coords = get_atom_coords(interaction.atom_2, interaction.res_2, df_atoms)
        if (
            closest_atom_1_coords is None
            or closest_atom_2_coords is None
            or atom_1_coords is None
            or atom_2_coords is None
        ):
            dihedral_angle = None
        else:
            dihedral_angle = calculate_dihedral_angle(
                closest_atom_1_coords,
                atom_1_coords,
                atom_2_coords,
                closest_atom_2_coords,
            )
        hbond_type = ""
        if interaction.res_1.get_str() + "-" + interaction.res_2.get_str() in pair_dict:
            pair = pair_dict[
                interaction.res_1.get_str() + "-" + interaction.res_2.get_str()
            ]
            if pair.bp_type == "WC":
                hbond_type = "BP-WC"
            else:
                hbond_type = "BP-NON-WC"
        elif interaction.atom_type_1 == "aa" or interaction.atom_type_2 == "aa":
            hbond_type = "RNA/PROTEIN"
        else:
            atom_1_type = interaction.atom_type_1
            atom_2_type = interaction.atom_type_2
            if atom_1_type > atom_2_type:
                atom_1_type, atom_2_type = atom_2_type, atom_1_type
            hbond_type = f"{atom_1_type}/{atom_2_type}"
        hbond_info = Hbond(
            interaction.res_1,
            interaction.res_2,
            interaction.atom_1,
            interaction.atom_2,
            interaction.atom_type_1,
            interaction.atom_type_2,
            interaction.distance,
            dihedral_angle,
            hbond_type,
            pdb_name,
        )
        hbonds.append(hbond_info)
    return hbonds
