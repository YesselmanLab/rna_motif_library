from rna_motif_library.hbond import HbondFactory
from rna_motif_library.resources import load_ideal_basepairs


def _test_hbond():
    basepairs = load_ideal_basepairs()
    bp = basepairs["AU_cWW"]
    hf = HbondFactory()
    hbond = hf.get_hbond(bp[0], bp[1], "N1", "N3")
    assert hbond is not None
    assert hbond.distance < 2.9
