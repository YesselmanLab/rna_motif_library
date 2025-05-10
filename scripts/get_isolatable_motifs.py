import pandas as pd
import os
from rna_motif_library.util import (
    get_pdb_ids,
    get_cached_path,
    get_nucleotide_atom_type,
    add_motif_name_columns,
)
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH


def get_isolatable_motifs():
    pdb_ids = get_pdb_ids()
    count = 0
    unique_motifs = pd.read_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv")
    )
    unique_motifs = unique_motifs["motif"].values
    data = []
    for pdb_id in pdb_ids:
        try:
            motifs = get_cached_motifs(pdb_id)
        except:
            continue
        for m in motifs:
            if m.name not in unique_motifs:
                continue
            if len(m.get_residues()) < 3:
                continue
            if m.mtype == "HELIX":
                continue
            res = []
            for r in m.get_residues():
                res.append(r.get_str())
            total = 0
            base_hbond = 0
            fail = False
            for hb in m.hbonds:
                # self hbond
                if hb.res_1.get_str() in res and hb.res_2.get_str() in res:
                    continue
                if hb.res_type_2 == "LIGAND":
                    fail = True
                    break
                if hb.res_type_2 == "SOLVENT":
                    continue
                total += 1
                atom_type = get_nucleotide_atom_type(hb.atom_1)
                if atom_type == "BASE":
                    base_hbond += 1
            if fail:
                continue
            if total > 5:
                continue
            if base_hbond > 1:
                continue
            count += 1
            data.append(m.name)
        print(count)
    df = pd.DataFrame({"motif": data})
    print(len(df))
    df.to_csv(
        os.path.join(DATA_PATH, "summaries", "isolatable_motifs.csv"), index=False
    )


def get_isolatable_motifs_summary():
    df = pd.read_csv(os.path.join(DATA_PATH, "summaries", "isolatable_motifs.csv"))
    df = add_motif_name_columns(df, "motif")
    data = []
    for j, h in df.groupby("msequence"):
        data.append({"count": len(h), "msequence": j, "repr_motif": h.iloc[0]["motif"]})
    df = pd.DataFrame(data)
    df.sort_values(by="count", ascending=False, inplace=True)
    df.to_csv(
        os.path.join(DATA_PATH, "summaries", "isolatable_motifs_summary.csv"),
        index=False,
    )


def main():
    get_isolatable_motifs_summary()


if __name__ == "__main__":
    main()
