"""
File for testing DSSR H-Bond functions


"""

import math
import os

from rna_motif_library.dssr import get_data_from_dssr, get_pdb_model_df
from rna_motif_library.dssr_hbonds import (
    assign_res_type,
    assemble_interaction_data,
    build_complete_hbond_interaction,
    merge_hbond_interaction_data,
    calc_distance,
    find_closest_atom,
    calculate_dihedral_angle,
)
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.snap import parse_snap_output


def test_assign_res_type() -> None:
    """
    Tests assigning residue types.

    Returns:
        None

    """
    list_of_residues = ["OP1", "O2'", "N1", "NZ"]
    list_of_residue_types = ["nt", "nt", "nt", "aa"]

    position = 0
    for item in list_of_residues:
        residue = item
        residue_type = list_of_residue_types[position]
        assigned_res_type = assign_res_type(residue, residue_type)
        if position == 0:
            assert assigned_res_type == "phos"
        elif position == 1:
            assert assigned_res_type == "sugar"
        elif position == 2:
            assert assigned_res_type == "base"
        elif position == 3:
            assert assigned_res_type == "aa"
        position += 1


def test_build_complete_hbond_interaction() -> None:
    """

    Builds complete interaction from JSON file and tests associated functionality.


    Returns:
        None

    """

    # import PDB
    name = "1GID"
    pdb_path = os.path.join(LIB_PATH, "resources", "1gid.cif")
    pdb_model_df = get_pdb_model_df(pdb_path)

    json_path = os.path.join(LIB_PATH, "resources", "1GID.json")
    motifs, hbonds = get_data_from_dssr(json_path)
    # ignore "motifs" as they are unused

    rnp_path = os.path.join(LIB_PATH, "resources", "1gid.out")

    unique_interaction_data = merge_hbond_interaction_data(
        parse_snap_output(out_file=rnp_path), hbonds
    )
    pre_assembled_interaction_data = assemble_interaction_data(unique_interaction_data)
    assembled_interaction_data = build_complete_hbond_interaction(
        pre_assembled_interaction_data, pdb_model_df, name
    )

    for interaction in assembled_interaction_data:
        # Check the class is loaded properly
        assert (
            str(type(interaction))
            == "<class 'rna_motif_library.classes.HBondInteraction'>"
        )

    # calc_distance test
    subset_df = pdb_model_df[pdb_model_df["id"] == 528]
    subset_df_2 = pdb_model_df[pdb_model_df["id"] == 529]
    distance = calc_distance(subset_df, subset_df_2)
    expected_distance = 1.490
    assert abs(distance - expected_distance) < 0.001

    # calculate_bond_angle
    # Load any four atoms from PDB, calculate their dihedral angle by hand to double check
    first_atom = pdb_model_df[pdb_model_df["id"] == 531]
    second_atom = pdb_model_df[pdb_model_df["id"] == 1787]

    third_atom = find_closest_atom(first_atom, pdb_model_df)
    fourth_atom = find_closest_atom(second_atom, pdb_model_df)
    bond_angle = calculate_dihedral_angle(
        first_atom, second_atom, third_atom, fourth_atom
    )
    expected_angle = 7.10
    assert abs(bond_angle - expected_angle) < 0.1


def main():
    test_assign_res_type()
    test_build_complete_hbond_interaction()


if __name__ == "__main__":
    main()
