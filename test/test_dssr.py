import os

from rna_motif_library.classes import Motif, Residue
from rna_motif_library.motif import get_motifs

RESOURCE_PATH = "test/resources"


def _test_residue_motif_rebuild_from_dict():
    motifs = get_motifs(
        1,
        os.path.join(RESOURCE_PATH, "1A9N.cif"),
    )
    m = motifs[0]
    # check that the residues can be rebuilt from the dictionary
    r: Residue = m.strands[0][0]
    r_new = Residue.from_dict(r.to_dict())
    assert r == r_new
    # check that the motif can be rebuilt from the dictionary
    m_new = Motif.from_dict(m.to_dict())
    assert m == m_new
