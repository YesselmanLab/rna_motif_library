import pandas as pd
import os

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import get_pdb_ids, get_non_redundant_sets
from rna_motif_library.tranforms import superimpose_structures, rmsd


def get_non_end_basepairs(motif):
    """Get all basepairs in a motif that are not end basepairs.

    Args:
        motif: Motif object to analyze

    Returns:
        List of Basepair objects that are not end basepairs
    """
    end_bps = motif.basepair_ends
    other_bps = []
    for bp in motif.basepairs:
        if bp not in end_bps:
            other_bps.append(bp)
    return other_bps


def get_tc_hbonds(pdb_id):
    try:
        tc_hbonds = pd.read_csv(
            os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
        )
    except:
        return {}
    tc_dict = {}
    for _, row in tc_hbonds.iterrows():
        if row["res_1"] not in tc_dict:
            tc_dict[row["res_1"]] = []
        if row["res_2"] not in tc_dict:
            tc_dict[row["res_2"]] = []
        tc_dict[row["res_1"]].append(
            [row["res_1"], row["res_2"], row["atom_1"], row["atom_2"], row["score"]]
        )
        tc_dict[row["res_2"]].append(
            [row["res_2"], row["res_1"], row["atom_2"], row["atom_1"], row["score"]]
        )
    return tc_dict


def get_motifs_summary(pdb_id):
    motifs = get_cached_motifs(pdb_id)
    tc_dict = get_tc_hbonds(pdb_id)

    data = []
    for m in motifs:
        res = []
        for r in m.get_residues():
            res.append(r.get_str())
        bps = get_non_end_basepairs(m)
        bps_info = []
        if m.mtype != "HELIX":
            for bp in bps:
                bps_info.append([bp.res_1.get_str(), bp.res_2.get_str(), bp.lw])
        prot_hbonds = []
        sm_hbonds = []
        for hb in m.hbonds:
            if hb.res_type_2 == "PROTEIN" or hb.res_type_2 == "NON CANONICAL AA":
                prot_hbonds.append(
                    [
                        hb.res_1.get_str(),
                        hb.res_2.get_str(),
                        hb.atom_1,
                        hb.atom_2,
                        hb.score,
                    ]
                )
            elif hb.res_type_2 == "LIGAND":
                sm_hbonds.append(
                    [
                        hb.res_1.get_str(),
                        hb.res_2.get_str(),
                        hb.atom_1,
                        hb.atom_2,
                        hb.score,
                    ]
                )
        tc_hbonds = []
        for res in m.get_residues():
            if res.get_str() in tc_dict:
                tc_hbonds.append(tc_dict[res.get_str()])
        data.append(
            {
                "name": m.name,
                "sequence": m.sequence,
                "size": m.size,
                "mtype": m.mtype,
                "residues": res,
                "non_canon_bps": bps_info,
                "prot_hbonds": prot_hbonds,
                "sm_hbonds": sm_hbonds,
                "tc_hbonds": tc_hbonds,
                "num_prot_hbonds": len(prot_hbonds),
                "num_sm_hbonds": len(sm_hbonds),
                "num_tc_hbonds": len(tc_hbonds),
            }
        )
    df = pd.DataFrame(data)
    return df


def get_all_motifs_summary():
    pdb_ids = get_pdb_ids()
    for pdb_id in pdb_ids:
        df = get_motifs_summary(pdb_id)
        df.sort_values(by="num_tc_hbonds", ascending=False, inplace=True)
        path = os.path.join(DATA_PATH, "dataframes", "motifs", f"{pdb_id}.json")
        df.to_json(path, orient="records")


def get_pdb_ids(df):
    pdb_ids = []
    for i, row in df.iterrows():
        spl = row["dup_motif"].split("-")
        pdb_ids.append(spl[-2])
    return pdb_ids


def main():
    # df = remove_duplicates()
    # df.to_csv("duplicates.csv", index=False)
    df = pd.read_csv("duplicates.csv")
    df["pdb_id"] = get_pdb_ids(df)
    for i, g in df.groupby("pdb_id"):
        print(i, len(g))
        path = os.path.join(DATA_PATH, "dataframes", "duplicate_motifs/", f"{i}.csv")
        g.to_csv(path, index=False)


if __name__ == "__main__":
    main()
