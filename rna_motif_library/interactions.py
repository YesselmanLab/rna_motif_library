import os
from typing import List, Tuple
import pandas as pd
import numpy as np

from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.classes import (
    X3DNAInteraction,
    HbondInfo,
    HBondInteraction,
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
    pdb_df_path = os.path.join(LIB_PATH, "data/pdbs_dfs", f"{pdb_name}.parquet")
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
                atom_1, res_1, atom_2, res_2, hbond.distance, type_1, type_2
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
        # Only keep interactions that aren't between two proteins
        is_protein_protein = (
            interaction.atom_type_1 == "aa" and interaction.atom_type_2 == "aa"
        )
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)


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


def get_angle(atom_1_coords, atom_2_coords, closest_atom_coords):
    v1 = np.array(atom_1_coords) - np.array(closest_atom_coords)
    v2 = np.array(atom_2_coords) - np.array(atom_1_coords)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_hbonds(
    pdb_name: str,
    df_atoms: pd.DataFrame,
    x3dna_interactions: List[X3DNAInteraction],
    x3dna_pairs: List[X3DNAPair],
):
    closest_atoms = get_closest_atoms()
    for interaction in x3dna_interactions:
        key = interaction.res_1.res_id + "-" + interaction.atom_1
        if key not in closest_atoms:
            log.warning(f"No closest atom found for {key}")
            closest_atom = None
            closest_atom_coords = None
        else:
            closest_atom = closest_atoms[key]
            closest_atom_coords = get_atom_coords(
                closest_atom, interaction.res_1, df_atoms
            )
        atom_1_coords = get_atom_coords(interaction.atom_1, interaction.res_1, df_atoms)
        atom_2_coords = get_atom_coords(interaction.atom_2, interaction.res_2, df_atoms)


def build_complete_hbond_interaction(
    interactions: List[X3DNAInteraction],
    pdb_name: str,
    df_atoms: pd.DataFrame,
) -> List[HbondInfo]:
    """
    Builds a complete HBondInteraction object from HBondInteractionFactory preliminary data

    Args:
        pre_assembled_interaction_data (list): pre-assembled HBondInteractionFactory data to draw data from
        pdb_model_df (pd.DataFrame): PDB dataframe
        pdb_name (str): name of source PDB

    Returns:
        built_interactions (list): a list of built HBondInteraction objects
    """

    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]

    built_interactions = []
    for interaction in interactions:
        key = interaction.res_1.res_id + "-" + interaction.atom_1
        if key not in closest_atoms:
            log.warning(f"No closest atom found for {key}")
            closest_atom = None
            closest_atom_coords = None
        else:
            closest_atom = closest_atoms[key]
            closest_atom_coords = get_atom_coords(
                closest_atom,
                interaction.res_1.res_id,
                interaction.res_1.num,
                interaction.res_1.chain_id,
                df_atoms,
            )
        atom_1_coords = get_atom_coords(
            interaction.atom_1,
            interaction.res_1.res_id,
            interaction.res_1.num,
            interaction.res_1.chain_id,
            df_atoms,
        )
        atom_2_coords = get_atom_coords(
            interaction.atom_2,
            interaction.res_2.res_id,
            interaction.res_2.num,
            interaction.res_2.chain_id,
            df_atoms,
        )
        # Skip if any coordinates are missing
        if (
            closest_atom_coords is None
            or atom_1_coords is None
            or atom_2_coords is None
        ):
            angle_degrees = None
        else:
            # Calculate vectors between points
            v1 = np.array(atom_1_coords) - np.array(closest_atom_coords)
            v2 = np.array(atom_2_coords) - np.array(atom_1_coords)

            # Calculate angle using dot product formula
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_degrees = np.degrees(angle)

        hbond_info = HbondInfo(
            interaction.res_1.res_id,
            interaction.res_2.res_id,
            interaction.atom_1,
            interaction.atom_2,
            interaction.atom_type_1,
            interaction.atom_type_2,
            interaction.distance,
            angle_degrees,
            pdb_name,
        )
        built_interactions.append(hbond_info)

    return built_interactions
