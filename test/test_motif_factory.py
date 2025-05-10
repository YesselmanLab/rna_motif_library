import pandas as pd

from rna_motif_library.motif_factory import (
    HairpinFinder,
    get_pdb_structure_data,
    get_cww_basepairs,
)

TEST_RESOURCES = "test/resources/motifs"


def test_hairpin_finder():
    pdb_data = get_pdb_structure_data("1GID")
    cww_basepairs = get_cww_basepairs(pdb_data)
    hf = HairpinFinder(pdb_data, cww_basepairs)
    hairpins = hf.get_hairpins()
    df_motifs = pd.read_json(TEST_RESOURCES + "/1GID.json")
    df_motifs = df_motifs[df_motifs["mtype"] == "HAIRPIN"]
    assert len(hairpins) == len(df_motifs)
    seen = []
    for i, row in df_motifs.iterrows():
        res = row["residues"]
        for m in hairpins:
            if m in seen:
                continue
            if row["sequence"] != m.sequence:
                continue
            m_res = [r.get_str() for r in m.get_residues()]
            if res == m_res:
                seen.append(m)
                break
    assert len(seen) == len(df_motifs)
