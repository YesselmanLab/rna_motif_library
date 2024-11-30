import os
from typing import List, Tuple
import pandas as pd

from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.classes import (
    X3DNAInteraction,
    Hbond,
    HBondInteraction,
    X3DNAResidue,
    X3DNAResidueFactory,
    canon_amino_acid_list,
)
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.settings import LIB_PATH


def get_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
    # build_complete_hbond_interaction(all_interactions, pdb_name)
    return all_interactions


def convert_hbonds_to_interactions(hbonds: List[DSSR_HBOND]) -> List[X3DNAInteraction]:
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
        type_1 = get_atom_type(atom_1)
        type_2 = get_atom_type(atom_2)
        for aa in canon_amino_acid_list:
            if aa in res_1:
                type_1 = "aa"
            if aa in res_2:
                type_2 = "aa"

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


def get_atom_coords(
    atom_name: str, res_name: str, chain_id: str, df: pd.DataFrame
) -> Tuple[float, float, float]:
    df_atom = df[
        (df["auth_atom_id"] == atom_name)
        & (df["auth_comp_id"] == res_name)
        & (df["auth_asym_id"] == chain_id)
    ]
    if df_atom.empty:
        return None
    return df_atom[["Cartn_x", "Cartn_y", "Cartn_z"]].values[0]


def build_complete_hbond_interaction(
    interactions: List[X3DNAInteraction],
    pdb_name: str,
) -> List[HBondInteraction]:
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
        closest_atom_1 = closest_atoms[
            interaction.res_1.res_id + "-" + interaction.atom_1
        ]
        closest_atom_2 = closest_atoms[
            interaction.res_2.res_id + "-" + interaction.atom_2
        ]
        print(closest_atom_1, closest_atom_2)
        exit()
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
