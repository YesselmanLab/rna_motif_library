import pandas as pd
import os
from typing import List

from rna_motif_library.motif import get_cached_motifs, Motif
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import get_pdb_ids, NonRedundantSetParser, NRSEntry
from rna_motif_library.tranforms import superimpose_structures, rmsd


def check_residue_numbers(motif1, motif2):
    """Compare residue numbers between two motifs.

    Args:
        motif1: First motif object
        motif2: Second motif object

    Returns:
        bool: True if motifs have same residue numbers, False otherwise
    """
    res1_nums = sorted([r.num for r in motif1.get_residues()])
    res2_nums = sorted([r.num for r in motif2.get_residues()])

    if len(res1_nums) != len(res2_nums):
        return False

    return res1_nums == res2_nums


def check_for_self_duplicates(motifs, other_motifs):
    data = []
    for i, m in enumerate(motifs):
        for j, om in enumerate(other_motifs):
            if m.name == om.name:
                continue
            if i > j:
                continue
            if m.sequence != om.sequence:
                continue
            if not check_residue_numbers(m, om):
                continue
            try:
                coords_1 = m.get_phos_coords()
                if len(coords_1) < 2:
                    continue
                coords_2 = om.get_phos_coords()
                if len(coords_2) != len(coords_1):
                    continue
                rotated_coords_2 = superimpose_structures(coords_2, coords_1)
                rmsd_val = rmsd(coords_1, rotated_coords_2)
                if rmsd_val < 0.10 * len(coords_1):
                    data.append(
                        {
                            "org_motif": m.name,
                            "dup_motif": om.name,
                            "rmsd": rmsd_val,
                        }
                    )

            except:
                print("issues", m.name, om.name)
                continue
    df = pd.DataFrame(data)
    return df


def check_for_duplicates(motifs, other_motifs):
    data = []
    used_motifs = []
    for i, om in enumerate(other_motifs):
        best_repr = None
        best_rmsd = 1000
        coords_1 = om.get_phos_coords()
        for j, m in enumerate(motifs):
            if m.name in used_motifs:
                continue
            if m.sequence != om.sequence:
                continue
            try:
                if len(coords_1) < 2:
                    continue
                coords_2 = m.get_phos_coords()
                if len(coords_2) != len(coords_1):
                    continue
                rotated_coords_2 = superimpose_structures(coords_2, coords_1)
                rmsd_val = rmsd(coords_1, rotated_coords_2)
                if rmsd_val < best_rmsd:
                    best_rmsd = rmsd_val
                    best_repr = m
            except:
                print("issues", m.name, om.name)
                continue
        is_duplicate = False
        if best_rmsd < 0.20 * len(coords_1):
            is_duplicate = True
        best_repr_name = best_repr.name if best_repr is not None else None
        data.append(
            {
                "motif": om.name,
                "repr_motif": best_repr_name,
                "rmsd": best_rmsd,
                "is_duplicate": is_duplicate,
            }
        )
        if is_duplicate:
            used_motifs.append(best_repr)

    df = pd.DataFrame(data)
    return df


def get_motifs(pdb_id: str):
    try:
        motifs = get_cached_motifs(pdb_id)
    except:
        print("missing motifs", pdb_id)
        return []
    return motifs


def get_entry_motifs(motifs: List[Motif], entry: NRSEntry):
    keep_motifs = []
    for m in motifs:
        keep = True
        for r in m.get_residues():
            if r.chain_id not in entry.chain_ids:
                keep = False
                break
        if keep:
            keep_motifs.append(m)
    return keep_motifs


def remove_duplicates():
    parser = NonRedundantSetParser()
    sets = parser.parse(os.path.join(DATA_PATH, "csvs", "nrlist_3.262_3.5A.csv"))
    for set_id, repr_entry, child_entries in sets:
        print(set_id)
        if len(child_entries) == 0:
            continue
        all_repr_motifs = get_motifs(repr_entry.pdb_id)
        repr_motifs = get_entry_motifs(all_repr_motifs, repr_entry)
        dfs = []
        for child_entry in child_entries:
            all_other_motifs = get_motifs(child_entry.pdb_id)
            other_motifs = get_entry_motifs(all_other_motifs, child_entry)
            df = check_for_duplicates(repr_motifs, other_motifs)
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(
            os.path.join(
                DATA_PATH,
                "dataframes",
                "duplicate_motifs",
                f"{set_id}.csv",
            ),
            index=False,
        )


def main():
    remove_duplicates()


if __name__ == "__main__":
    main()
