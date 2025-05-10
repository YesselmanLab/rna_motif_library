import click
import pandas as pd

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_factory import (
    HelixFinder,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    get_cww_basepairs,
)


def main():
    df = pd.read_csv("large_motifs.csv")
    df["num_helices"] = 0
    count = 0
    for pdb_id, g in df.groupby("pdb_id"):
        motifs = get_cached_motifs(pdb_id)
        pdb_data = get_pdb_structure_data(pdb_id)
        cww_basepairs = get_cww_basepairs(
            pdb_data, min_two_hbond_score=0.5, min_three_hbond_score=0.5
        )
        motif_by_name = {m.name: m for m in motifs}
        for i, row in g.iterrows():
            m = motif_by_name[row["motif_name"]]
            res = m.get_residues()
            pdb_data_for_residues = get_pdb_structure_data_for_residues(pdb_data, res)
            hf = HelixFinder(pdb_data_for_residues, cww_basepairs, [])
            helices = hf.get_helices()
            df.loc[i, "num_helices"] = len(helices)
            if len(helices) > 1:
                print(pdb_id, row["motif_name"], len(helices), count)
                count += 1


if __name__ == "__main__":
    main()
