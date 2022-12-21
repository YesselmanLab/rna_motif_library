import pytest

from rna_motif_library import snap, settings

def test_snap():
    pdb_path = settings.UNITTEST_PATH + "/resources/4b3g.pdb"
    interactions = snap.get_rnp_interactions(pdb_path)
    assert len(interactions) == 43


if __name__ == "__main__":
    test_snap()
