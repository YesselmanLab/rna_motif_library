from rna_motif_library.hbond import HbondFactory
from rna_motif_library.resources import load_ideal_basepairs


def test_get_hbond():
    basepairs = load_ideal_basepairs()
    bp = basepairs["AU_cWW"]
    res = list(bp.values())
    hf = HbondFactory()
    hbond = hf.get_hbond(res[0], res[1], "N1", "N3", "TEST")
    assert hbond is not None
    assert hbond.distance < 3.0


def test_find_hbonds():
    basepairs = load_ideal_basepairs()
    bp = basepairs["AU_cWW"]
    res = list(bp.values())
    hf = HbondFactory()
    hbonds = hf.find_hbonds(res[0], res[1], "TEST")
    assert len(hbonds) == 2
