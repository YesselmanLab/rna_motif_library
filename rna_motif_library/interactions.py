import os
from typing import List, Tuple
from dataclasses import dataclass


from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.classes import X3DNAInteraction
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.settings import LIB_PATH


def get_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
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

        # Swap if res_1 is larger than res_2
        if res_1 > res_2:
            atom_1, atom_2 = atom_2, atom_1
            res_1, res_2 = res_2, res_1
            type_1, type_2 = type_2, type_1

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
        is_protein_protein = interaction.type_1 == "aa" and interaction.type_2 == "aa"
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)
