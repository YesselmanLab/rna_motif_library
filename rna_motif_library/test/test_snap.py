from rna_motif_library import settings
from rna_motif_library.snap import parse_snap_output


def test_snap() -> None:
    """
    Tests the retrieval of interactions from SNAP

    Returns:
        None

    """
    pdb_path = str(settings.UNITTEST_PATH) + "/resources/4b3g.pdb"
    out_path = str(settings.UNITTEST_PATH) + "/resources/4b3g.out"
    interactions = parse_snap_output(pdb_path, out_path)
    assert len(interactions) == 43


if __name__ == "__main__":
    test_snap()
