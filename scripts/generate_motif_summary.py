import pandas as pd
import os

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import (
    get_pdb_ids,
    get_non_redundant_sets,
    add_motif_indentifier_columns,
)
from rna_motif_library.dataframe_tools import get_cif_str_from_row
from rna_motif_library.tranforms import superimpose_structures, rmsd
from rna_motif_library.parallel_utils import run_w_processes_in_batches
from rna_motif_library.util import get_nucleotide_atom_type, canon_rna_res_list


def get_basepairs_that_contain_residues(df, residue_ids):
    """Filter basepairs dataframe to only include rows where both residues are in the given list.

    Args:
        df: DataFrame containing basepair information with res_1 and res_2 columns
        residue_ids: List of residue identifiers to filter by

    Returns:
        DataFrame containing only rows where both res_1 and res_2 are in residue_ids
    """
    return df[(df["res_1"].isin(residue_ids)) & (df["res_2"].isin(residue_ids))]


def get_basepairs_that_do_not_contain_residues(df, residue_ids):
    """Filter basepairs dataframe to only include rows where both residues are in the given list.

    Args:
        df: DataFrame containing basepair information with res_1 and res_2 columns
        residue_ids: List of residue identifiers to filter by

    Returns:
        DataFrame containing only rows where both res_1 and res_2 are in residue_ids
    """
    return df[~((df["res_1"].isin(residue_ids)) & (df["res_2"].isin(residue_ids)))]


def get_non_canonical_basepair_summary(m, df_bps):
    if m.mtype == "HELIX":
        return []
    if len(df_bps) == 0:
        return []
    res = [r.get_str() for r in m.get_residues()]
    chain_end_res = []
    for strand in m.strands:
        chain_end_res.extend([r.get_str() for r in [strand[0], strand[-1]]])
    df_bps_sub = get_basepairs_that_contain_residues(df_bps, res)
    df_bps_sub = get_basepairs_that_do_not_contain_residues(df_bps_sub, chain_end_res)
    bps = []
    for _, row in df_bps_sub.iterrows():
        bps.append([row["res_1"], row["res_2"], row["lw"], row["hbond_score"]])
    return bps


def is_motif_isolatable(m):
    if len(m.get_residues()) < 3:
        return False
    if m.mtype == "HELIX":
        return False
    res = []
    for r in m.get_residues():
        res.append(r.get_str())
    total = 0
    base_hbond = 0
    fail = 0
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
        return 0
    if total > 5:
        return 0
    if base_hbond > 1:
        return 0
    return 1


def get_tertiary_contact_summary(m, df_tc):
    if len(df_tc) == 0:
        return {
            "num_tertiary_contacts": 0,
            "num_hbonds": 0,
            "hbond_score": 0,
            "in_tertiary_contact": 0,
        }
    df_sub = df_tc.query("motif_1 == @m.name or motif_2 == @m.name")
    return {
        "num_tertiary_contacts": len(df_sub),
        "num_hbonds": df_sub["num_hbonds"].sum(),
        "hbond_score": df_sub["hbond_score"].sum(),
        "in_tertiary_contact": 1 if len(df_sub) > 0 else 0,
    }


def get_motifs_summary(args):
    pdb_id = args[0]
    df = args[1]
    path = os.path.join(DATA_PATH, "dataframes", "motifs", f"{pdb_id}.json")
    if os.path.exists(path):
        return pd.read_json(path)
    path = os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json")
    if not os.path.exists(path):
        df_tc = pd.DataFrame()
    else:
        df_tc = pd.read_json(path)
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {m.name: m for m in motifs}
    df_bps = pd.read_json(
        os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_id}.json")
    )

    data = []
    for i, row in df.iterrows():
        m = motifs_by_name[row["motif_name"]]
        res = [r.get_str() for r in m.get_residues()]
        atom_names = []
        coords = []
        for r in m.get_residues():
            atom_names.append(r.atom_names)
            coords.append(r.coords)
        non_canonical_bps = get_non_canonical_basepair_summary(m, df_bps)
        tc_summary = get_tertiary_contact_summary(m, df_tc)
        prot_hbond_score = 0
        num_prot_hbonds = 0
        sm_hbond_score = 0
        num_sm_hbonds = 0
        for hb in m.hbonds:
            if hb.res_type_2 == "PROTEIN" or hb.res_type_2 == "NON CANONICAL AA":
                prot_hbond_score += hb.score
                num_prot_hbonds += 1
            elif hb.res_type_2 == "LIGAND":
                sm_hbond_score += hb.score
                num_sm_hbonds += 1
        has_non_canonical_basepair_flank = 0
        for bp in m.basepair_ends:
            if (
                bp.res_1.res_id not in canon_rna_res_list
                or bp.res_2.res_id not in canon_rna_res_list
            ):
                has_non_canonical_basepair_flank = 1
                break
        data.append(
            {
                "pdb_id": pdb_id,
                "motif_id": m.name,
                "motif_sequence": m.sequence,
                "motif_topology": m.size,
                "motif_type": m.mtype,
                "residues": res,
                "num_non_canonical_basepairs": len(non_canonical_bps),
                "num_strands": len(m.strands),
                "num_residues": len(res),
                "non_canonical_bps": non_canonical_bps,
                "num_hbonds": len(m.hbonds),
                "num_protein_hbonds": num_prot_hbonds,
                "num_ligand_hbonds": num_sm_hbonds,
                "num_tc_hbonds": tc_summary["num_hbonds"],
                "protein_hbond_score": prot_hbond_score,
                "ligand_hbond_score": sm_hbond_score,
                "tertiary_contact_hbond_score": tc_summary["hbond_score"],
                "has_singlet_pair": row["has_singlet_pair"],
                "has_non_canonical_residue": int(m.sequence.count("X") > 0),
                "has_non_canonical_basepair_flank": has_non_canonical_basepair_flank,
                "is_isolatable": is_motif_isolatable(m),
                "in_tertiary_contact": tc_summary["in_tertiary_contact"],
                "num_tertiary_contacts": tc_summary["num_tertiary_contacts"],
                "atom_names": atom_names,
                "coords": coords,
                "unique": 1,
            }
        )
    df = pd.DataFrame(data)
    df.to_json(
        os.path.join(DATA_PATH, "dataframes", "motifs", f"{pdb_id}.json"),
        orient="records",
    )
    return df


def main():
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "motifs"), exist_ok=True)
    path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    df = pd.read_csv(path)
    df = add_motif_indentifier_columns(df, "motif_name")

    # Group by PDB ID and process in parallel
    groups = list(df.groupby("pdb_id"))

    results = run_w_processes_in_batches(
        items=groups,
        func=get_motifs_summary,
        processes=10,  # Use 10 processes
        batch_size=100,  # Process 100 PDBs at a time
        desc="Processing PDB IDs for motif summaries",
    )

    # Combine results
    dfs = [df for df in results if df is not None]
    df = pd.concat(dfs)
    df.to_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "release",
            "motifs",
            "non_redundant_motif_summary_w_coords.json",
        ),
        orient="records",
    )
    df = df.drop(columns=["atom_names", "coords"])
    df.to_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "release",
            "motifs",
            "non_redundant_motif_summary.json",
        ),
        orient="records",
    )
    os.system(
        "gzip -9 {}".format(
            os.path.join(
                DATA_PATH,
                "summaries",
                "release",
                "motifs",
                "non_redundant_motif_summary.json",
            )
        )
    )
    os.system(
        "gzip -9 {}".format(
            os.path.join(
                DATA_PATH,
                "summaries",
                "release",
                "motifs",
                "non_redundant_motif_summary_w_coords.json",
            )
        )
    )


if __name__ == "__main__":
    main()
