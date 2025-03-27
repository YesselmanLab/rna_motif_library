import pandas as pd
import os
from rna_motif_library.util import (
    get_pdb_ids,
    get_cached_path,
    get_nucleotide_atom_type,
)
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH


def main():
    pdb_ids = get_pdb_ids()
    count = 0
    for pdb_id in pdb_ids:
        if not os.path.exists(get_cached_path(pdb_id, "motifs")):
            continue
        dup_motifs = []
        path = os.path.join(
            DATA_PATH, "dataframes", "duplicate_motifs", f"{pdb_id}.csv"
        )
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dup_motifs = df["dup_motif"].values
            except Exception as e:
                dup_motifs = []
                continue
        motifs = get_cached_motifs(pdb_id)
        for m in motifs:
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
            print(pdb_id, m.name, total, base_hbond, count)
            count += 1
        print(count)


if __name__ == "__main__":
    main()
