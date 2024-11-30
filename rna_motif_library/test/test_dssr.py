import os
import pandas as pd

from rna_motif_library.classes import X3DNAResidue, extract_longest_numeric_sequence
from rna_motif_library.dssr import (
    find_strands,
    get_data_from_dssr,
    determine_motif_type,
)
from rna_motif_library.settings import LIB_PATH


def test_extract_num_seq() -> None:
    """
    Tests extracting longest numeric sequences; used in DSSR parsing.

    Returns:
        None

    """
    seq1 = "A23F4523D1"
    assert extract_longest_numeric_sequence(seq1) == "4523"
    seq2 = "E.123.45-72"
    assert extract_longest_numeric_sequence(seq2) == "123"
    seq3 = "F.45.23"
    assert extract_longest_numeric_sequence(seq3) == "23"
    # In the event of a tie it should take the latest sequence; the latest will refer to the residue ID in a PDB.


def test_dssr_res() -> None:
    """
    Tests the parsing of resides from DSSR data.

    Returns:
        None
    """
    s1 = "H.A9"
    s2 = "B.ARG270"
    r1 = X3DNAResidue(s1)
    assert r1.res_id == "A"
    assert r1.chain_id == "H"
    assert r1.num == 9
    r2 = X3DNAResidue(s2)
    assert r2.res_id == "ARG"
    assert r2.num == 270


def test_find_strands_sequence() -> None:
    """
    Tests counting strands and finding sequences.

    Returns:
        None
    """
    path = os.path.join(LIB_PATH, "resources", "find_strands")

    # Test helix
    pdb_df_helix = import_cif_as_dataframe(
        os.path.join(path, "HELIX.5JUP.2.AU-GU.0.cif")
    )
    list_of_strands, sequence = find_strands(pdb_df_helix)
    assert len(list_of_strands) == 2
    assert sequence == "AU-GU"

    # Test nway
    pdb_df_nway = import_cif_as_dataframe(
        os.path.join(path, "NWAY.2BTE.2-2-9-2-4.CG-UG-GCAAGCGUG-CC-GUGG.0.cif")
    )
    list_of_strands, sequence = find_strands(pdb_df_nway)
    assert len(list_of_strands) == 5
    assert sequence == "CG-UG-GCAAGCGUG-CC-GUGG"


def test_determine_motif_type() -> None:
    """
    Tests determining motif types, given the DSSR JSON output.

    Returns:
        None

    """
    # Load the desired JSON data
    json_path = os.path.join(LIB_PATH, "resources", "1GID.json")
    motifs, hbonds = get_data_from_dssr(json_path)
    # we don't need hbonds so this variable will just remain unused

    for m in motifs:
        motif_type = determine_motif_type(m)

        if m.mtype in ["JUNCTION", "BULGE", "ILOOP"]:
            assert motif_type == "JCT"
        elif m.mtype in ["STEM", "HEXIX"]:
            assert motif_type == "HELIX"
        elif m.mtype in ["SINGLE_STRAND"]:
            assert motif_type == "SSTRAND"
        elif m.mtype in ["HAIRPIN"]:
            assert motif_type == "HAIRPIN"
        else:
            continue


def import_cif_as_dataframe(cif_path: str) -> pd.DataFrame:
    """
    Imports a tabular .cif file as a pandas DataFrame.
    Temporary function to import CIFs for testing.

    Args:
        cif_path (str): Path to the .cif file.

    Returns:
        pd.DataFrame: DataFrame containing the tabular data from the .cif file.

    """
    with open(cif_path, "r") as file:
        lines = file.readlines()

    # Identify the start of the table (e.g., loop_ keyword in the CIF file)
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("loop_"):
            start_idx = idx
            break

    if start_idx is None:
        raise ValueError("No tabular data (loop_) found in the .cif file.")

    # Extract column names and remove prefixes
    columns = []
    for line in lines[start_idx + 1 :]:
        if line.startswith("_"):
            # Remove everything before and including the last dot
            columns.append(line.strip().split(".")[-1])
        else:
            break

    # Extract the data
    data = []
    for line in lines[start_idx + 1 + len(columns) :]:
        if line.startswith("_") or line.strip() == "":
            break
        data.append(line.strip().split())

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    return df


def main():
    test_dssr_res()
    test_extract_num_seq()
    test_find_strands_sequence()
    test_determine_motif_type()


if __name__ == "__main__":
    main()
