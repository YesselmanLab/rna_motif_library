"""
File for testing DSSR H-Bond functions


"""
from rna_motif_library.dssr_hbonds import assign_res_type


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
    # Load an example of an intermediate class and use it plus JSON data to build the complete HBondInteraction object

    # Also tests:
    # calculate_bond_angle
    # Load any four atoms, calculate their dihedral angle by hand to double check
    # calc_distance
    # Load any two random atoms here and calculate their distance by hand to double check
    # find_closest_atom
    # Need to import raw interaction from JSON and use one interaction to do it
    # extract_interacting_atoms
    # Need to import raw interaction from JSON and use one interaction to do it

    pass


def main():
    pass


if __name__ == "__main__":
    main()
