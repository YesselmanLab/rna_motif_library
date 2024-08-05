import dssr

# test:
# extract_longest_numeric_sequence
#


def test_dssr_res() -> None:
    """
    Tests the parsing of resides from DSSR data

    Returns:
        None

    """
    s1 = "H.A9"
    s2 = "B.ARG270"
    r1 = dssr.DSSRRes(s1)
    assert r1.res_id == "A"
    assert r1.chain_id == "H"
    assert r1.num == 9
    r2 = dssr.DSSRRes(s2)
    assert r2.res_id == "ARG"
    assert r2.num == 270


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
        assigned_res_type = dssr.assign_res_type(residue, residue_type)
        if position == 0:
            assert assigned_res_type == "phos"
        elif position == 1:
            assert assigned_res_type == "sugar"
        elif position == 2:
            assert assigned_res_type == "base"
        elif position == 3:
            assert assigned_res_type == "aa"
        position += 1

    list_of_residues_2 = ["OP1", "O2'", "N1", "O1P"]
    position = 0
    for item in list_of_residues_2:
        assigned_res_type = dssr.__assign_atom_group(item)
        if position == 0:
            assert assigned_res_type == "phos"
        elif position == 1:
            assert assigned_res_type == "sugar"
        elif position == 2:
            assert assigned_res_type == "base"
        elif position == 3:
            assert assigned_res_type == "phos"
        position += 1


def main():
    test_dssr_res()
    test_assign_res_type()


if __name__ == "__main__":
    main()
