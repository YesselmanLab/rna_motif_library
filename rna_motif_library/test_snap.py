import settings
import snap


def test_snap():
    """
    Tests the retrieval of interactions from SNAP
    """
    pdb_path = str(settings.UNITTEST_PATH) + "resources/4b3g.pdb"
    out_path = str(settings.UNITTEST_PATH) + "resources/4b3g.out"
    interactions = snap.get_rnp_interactions(pdb_path, out_path)
    assert len(interactions) == 43


if __name__ == "__main__":
    test_snap()
