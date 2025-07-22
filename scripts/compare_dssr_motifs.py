import pandas as pd
import click
import os

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_analysis import MotifFactoryFromOther
from rna_motif_library.motif_factory import get_pdb_structure_data


def main():
    os.makedirs("overlapping_motifs_dssr", exist_ok=True)
    df = pd.read_json("overlapping_motifs.json")
    count = 0
    for pdb_id, g in df.groupby("pdb_id"):
        try:
            pdb_data = get_pdb_structure_data(pdb_id)
        except Exception as e:
            print(f"Error getting PDB data for {pdb_id}: {e}")
            continue
        mf = MotifFactoryFromOther(pdb_data)
        dssr_motifs = mf.get_motifs_from_dssr()
        dssr_motif_by_name = {m.name: m for m in dssr_motifs}
        motifs = get_cached_motifs(pdb_id)
        motif_by_name = {m.name: m for m in motifs}
        for _, row in g.iterrows():
            os.makedirs(
                os.path.join("overlapping_motifs_dssr", str(count)), exist_ok=True
            )
            dssr_motif = dssr_motif_by_name[row["motif"]]
            dssr_motif.to_cif(
                os.path.join(
                    "overlapping_motifs_dssr",
                    str(count),
                    f"DSSR_{row['motif']}.cif",
                )
            )
            for m_name in row["overlapping_motifs"]:
                motif = motif_by_name[m_name]
                motif.to_cif(
                    os.path.join("overlapping_motifs_dssr", str(count), f"{m_name}.cif")
                )
            count += 1
            if count > 10:
                exit()


if __name__ == "__main__":
    main()
