import pandas as pd

from rna_motif_library.motif import get_cached_motifs
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


# TODO potentially and columns for non-canonical basepairs
def main():
    data = []
    motifs = get_cached_motifs("7WIE")
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
                prot_hbonds.append([hb.res_1.get_str(), hb.res_2.get_str(), hb.atom_1, hb.atom_2, hb.score])
            elif hb.res_type_2 == "LIGAND":
                sm_hbonds.append([hb.res_1.get_str(), hb.res_2.get_str(), hb.atom_1, hb.atom_2, hb.score])
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
                "num_prot_hbonds": len(prot_hbonds),
                "num_sm_hbonds": len(sm_hbonds),
            }
        )
    df = pd.DataFrame(data)
    df.sort_values(by="num_tert_hbonds", inplace=True, ascending=False)
    for i, row in df.iterrows():
        for hb in row["tert_hbonds"]:
            print(hb)
        exit()
    df.to_json("motif_summary.json", orient="records")


if __name__ == "__main__":
    main()
