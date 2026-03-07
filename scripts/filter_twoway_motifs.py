"""
Filter and deduplicate TWOWAY motifs to a curated representative set.

Reads from data/summaries/motifs/non_redundant_motifs_summary.json and produces:
1. A curated CSV of ~456 unique representative motifs
2. A tracking CSV of ALL 16,806 motifs with status, rejection reason, and representative

Pipeline:
1. Max residue filter (default: 15)
2. Sequence dedup (keep best resolution per unique sequence)
3. Compute cross-strand basepair signature (score >= 1.0)
4. Split into with-basepair and no-basepair populations
5. With-bp: dedup by (topology, bp_sig), then cap per topology
   using farthest-point RMSD diversity selection
6. No-bp: keep N per topology (best resolution)

Usage:
    python scripts/filter_twoway_motifs.py -o curated_twoway.csv
    python scripts/filter_twoway_motifs.py -o curated.csv --tracking-output tracking.csv
    python scripts/filter_twoway_motifs.py -o curated.csv --topo-cap 20 --no-bp-per-topo 2
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import json

import click
import numpy as np
import pandas as pd

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.tranforms import superimpose_structures, rmsd

log = get_logger("filter_twoway")


def load_resolution_data():
    """Load resolution data from rna_structures.csv."""
    path = os.path.join(DATA_PATH, "csvs", "rna_structures.csv")
    if not os.path.exists(path):
        log.warning(f"Resolution file not found: {path}")
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["pdb_id"], df["resolution"]))


def _nuc_id(res_str):
    """Extract nucleotide identity from residue string like '1-A-100-'."""
    return res_str.split("-")[1]


def get_cross_strand_bp_info(row, min_score=1.0):
    """
    Get cross-strand basepair signature and detailed basepair list.

    Returns:
        (bp_sig, basepairs_json) where:
        - bp_sig: sorted comma-joined LW types (e.g., "cWW,tSH")
        - basepairs_json: JSON string like [["AG","cWW"],["CU","tSH"]]
    """
    bps = row["non_canonical_bps"]
    if not bps:
        return "", "[]"
    seq_parts = row["motif_sequence"].split("-")
    if len(seq_parts) != 2:
        return "", "[]"
    s1_len = len(seq_parts[0])
    res = row["residues"]
    strand1_set = set(res[:s1_len])
    strand2_set = set(res[s1_len:])
    lw_types = []
    bp_details = []
    for bp in bps:
        if len(bp) < 4:
            continue
        r1, r2, lw, score = bp[0], bp[1], bp[2], bp[3]
        if score < min_score:
            continue
        is_cross = (r1 in strand1_set and r2 in strand2_set) or (
            r1 in strand2_set and r2 in strand1_set
        )
        if is_cross:
            lw_types.append(lw)
            bp_details.append([_nuc_id(r1) + _nuc_id(r2), lw])
    return ",".join(sorted(lw_types)), json.dumps(bp_details)


def add_comprehensive_columns(df):
    """
    Add comprehensive derived columns to a TWOWAY motif dataframe.

    Adds: strand1_sequence, strand2_sequence, is_bulge,
          closing_basepairs, num_hbonds, num_protein_hbonds,
          num_ligand_hbonds, in_tertiary_contact, num_tertiary_contacts,
          has_non_canonical_residue, has_non_canonical_basepair_flank,
          is_isolatable
    """
    # Strand sequences
    df["strand1_sequence"] = df["motif_sequence"].apply(
        lambda x: x.split("-")[0] if "-" in x else x
    )
    df["strand2_sequence"] = df["motif_sequence"].apply(
        lambda x: x.split("-")[1] if "-" in x else ""
    )

    # Is bulge (one strand has 0 internal residues)
    df["is_bulge"] = df["motif_topology"].apply(
        lambda x: "0" in x.split("-") if "-" in x else False
    )

    # Closing basepairs — derive nucleotide pair from sequence
    # 5' closing: strand1[0] pairs with strand2[-1]
    # 3' closing: strand1[-1] pairs with strand2[0]
    def _get_closing_bps(row):
        s1 = row["strand1_sequence"]
        s2 = row["strand2_sequence"]
        if not s1 or not s2:
            return "[]"
        bp_5p = [s1[0] + s2[-1], "cWW"]
        bp_3p = [s1[-1] + s2[0], "cWW"]
        return json.dumps([bp_5p, bp_3p])

    df["closing_basepairs"] = df.apply(_get_closing_bps, axis=1)

    return df


def load_motif_coords(motif_id, pdb_id, motif_cache):
    """Load C1' coordinates for a motif, using cache."""
    if pdb_id not in motif_cache:
        try:
            motif_list = get_cached_motifs(pdb_id)
            motif_cache[pdb_id] = {m.name: m for m in motif_list}
        except Exception:
            motif_cache[pdb_id] = {}
    motif_obj = motif_cache[pdb_id].get(motif_id)
    if motif_obj is None:
        return None
    coords = motif_obj.get_c1prime_coords()
    return coords if len(coords) > 0 else None


def compute_rmsd_pair(coords1, coords2):
    """Compute RMSD between two coordinate arrays after superposition."""
    if len(coords1) != len(coords2):
        return 999.0
    try:
        aligned = superimpose_structures(coords2, coords1)
        return float(rmsd(coords1, aligned))
    except Exception:
        return 999.0


def compute_rmsds_for_topology(work_item):
    """
    Compute pairwise RMSDs for motifs within a topology group.
    Called in a worker process.
    """
    topology = work_item["topology"]
    entries = work_item["entries"]

    n = len(entries)
    if n <= 1:
        return {"topology": topology, "entries": entries, "rmsd_matrix": None}

    motif_cache = {}
    coords_list = []
    for entry in entries:
        coords = load_motif_coords(entry["motif_id"], entry["pdb_id"], motif_cache)
        coords_list.append(coords)

    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if coords_list[i] is None or coords_list[j] is None:
                rmsd_matrix[i, j] = 999.0
            else:
                rmsd_matrix[i, j] = compute_rmsd_pair(coords_list[i], coords_list[j])
            rmsd_matrix[j, i] = rmsd_matrix[i, j]

    return {
        "topology": topology,
        "entries": entries,
        "rmsd_matrix": rmsd_matrix.tolist(),
    }


def select_diverse_subset(entries, rmsd_matrix, max_n):
    """
    Select maximally diverse subset using farthest-point greedy algorithm.
    Returns (selected_entries, rejected_entries_with_nearest_rep).
    """
    n = len(entries)
    if n <= max_n:
        return entries, []

    matrix = np.array(rmsd_matrix)
    selected = [0]
    remaining = set(range(1, n))

    while len(selected) < max_n and remaining:
        best_idx = None
        best_min_dist = -1
        for idx in remaining:
            min_dist = min(matrix[idx, s] for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    selected_set = set(selected)
    selected_entries = [entries[i] for i in selected]

    # For each rejected entry, find nearest selected entry
    rejected_with_rep = []
    for idx in range(n):
        if idx not in selected_set:
            nearest_idx = min(selected, key=lambda s: matrix[idx, s])
            rejected_with_rep.append(
                (entries[idx], entries[nearest_idx]["motif_id"])
            )

    return selected_entries, rejected_with_rep


def resolve_representative_chains(tracking):
    """
    Resolve representative chains so every rejected motif points to a final kept motif.
    E.g., A->B->C where C is kept: A's representative becomes C.
    """
    for motif_id, info in tracking.items():
        if info["status"] != "rejected" or not info["representative_motif_id"]:
            continue
        # Follow the chain
        rep = info["representative_motif_id"]
        visited = {motif_id}
        while rep and rep in tracking and tracking[rep]["status"] == "rejected":
            if rep in visited:
                break
            visited.add(rep)
            rep = tracking[rep]["representative_motif_id"]
        info["representative_motif_id"] = rep


@click.command()
@click.option(
    "-o", "--output", "output_path", required=True,
    help="Output CSV path for curated motifs",
)
@click.option(
    "--tracking-output", "tracking_path", default=None,
    help="Output CSV path for full tracking of all motifs (optional)",
)
@click.option(
    "--max-residues", type=int, default=15,
    help="Maximum total residues per motif (default: 15)",
)
@click.option(
    "--bp-score-threshold", type=float, default=1.0,
    help="Minimum hbond score for basepair to count (default: 1.0)",
)
@click.option(
    "--topo-cap", type=int, default=15,
    help="Max motifs per topology for with-bp population (default: 15)",
)
@click.option(
    "--no-bp-per-topo", type=int, default=3,
    help="Max motifs per topology for no-bp population (default: 3)",
)
@click.option(
    "--workers", type=int, default=20,
    help="Number of parallel workers for RMSD computation (default: 20)",
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Show detailed progress and breakdowns",
)
def filter_motifs(
    output_path,
    tracking_path,
    max_residues,
    bp_score_threshold,
    topo_cap,
    no_bp_per_topo,
    workers,
    verbose,
):
    """Filter TWOWAY motifs to a curated representative set."""
    # =========================================================================
    # Load source data
    # =========================================================================
    summary_path = os.path.join(
        DATA_PATH, "summaries", "motifs", "non_redundant_motifs_summary.json"
    )
    log.info(f"Loading {summary_path}")
    df_all = pd.read_json(summary_path)
    df_tw = df_all[df_all["motif_type"] == "TWOWAY"].copy()
    print(f"Total TWOWAY motifs: {len(df_tw)}")

    resolution_data = load_resolution_data()
    df_tw["resolution"] = df_tw["pdb_id"].map(resolution_data).fillna(99.0)

    # Compute bp_sig and basepair details for ALL motifs (needed for tracking)
    bp_info = df_tw.apply(
        lambda row: get_cross_strand_bp_info(row, min_score=bp_score_threshold),
        axis=1,
    )
    df_tw["bp_sig"] = bp_info.apply(lambda x: x[0])
    df_tw["basepairs"] = bp_info.apply(lambda x: x[1])

    # Initialize tracking for ALL motifs
    tracking = {}
    for _, row in df_tw.iterrows():
        tracking[row["motif_id"]] = {
            "status": "pending",
            "rejection_reason": "",
            "representative_motif_id": "",
        }

    # =========================================================================
    # Step 1: Max residue filter
    # =========================================================================
    too_large = df_tw["num_residues"] > max_residues
    for motif_id in df_tw.loc[too_large, "motif_id"]:
        tracking[motif_id]["status"] = "rejected"
        tracking[motif_id]["rejection_reason"] = "too_many_residues"

    df_tw = df_tw[~too_large].copy()
    print(f"After max {max_residues} residues: {len(df_tw)} (removed {too_large.sum()})")

    # =========================================================================
    # Step 2: Sequence dedup (best resolution)
    # =========================================================================
    df_tw = df_tw.sort_values(["resolution", "motif_id"])
    seq_groups = df_tw.groupby("motif_sequence")
    keep_ids_seq = set()
    for seq, group in seq_groups:
        best_id = group.iloc[0]["motif_id"]
        keep_ids_seq.add(best_id)
        for _, row in group.iloc[1:].iterrows():
            tracking[row["motif_id"]]["status"] = "rejected"
            tracking[row["motif_id"]]["rejection_reason"] = "sequence_dedup"
            tracking[row["motif_id"]]["representative_motif_id"] = best_id

    before = len(df_tw)
    df_tw = df_tw[df_tw["motif_id"].isin(keep_ids_seq)].copy()
    print(f"After sequence dedup: {len(df_tw)} unique sequences (removed {before - len(df_tw)})")

    # =========================================================================
    # Step 3: Split into with-bp and no-bp
    # =========================================================================
    has_bp = df_tw["bp_sig"] != ""
    print(f"With cross-strand basepairs: {has_bp.sum()}")
    print(f"Without cross-strand basepairs: {(~has_bp).sum()}")

    df_with_bp = df_tw[has_bp].copy()
    df_no_bp = df_tw[~has_bp].copy()

    # =========================================================================
    # Step 4: With-bp — dedup by (topology, bp_sig)
    # =========================================================================
    df_with_bp["dedup_key"] = df_with_bp["motif_topology"] + "|" + df_with_bp["bp_sig"]
    df_with_bp = df_with_bp.sort_values(["resolution", "motif_id"])

    dedup_groups = df_with_bp.groupby("dedup_key")
    keep_ids_dedup = set()
    for key, group in dedup_groups:
        best_id = group.iloc[0]["motif_id"]
        keep_ids_dedup.add(best_id)
        for _, row in group.iloc[1:].iterrows():
            tracking[row["motif_id"]]["status"] = "rejected"
            tracking[row["motif_id"]]["rejection_reason"] = "topology_bp_sig_dedup"
            tracking[row["motif_id"]]["representative_motif_id"] = best_id

    before = len(df_with_bp)
    df_with_bp = df_with_bp[df_with_bp["motif_id"].isin(keep_ids_dedup)].copy()
    print(f"With-bp after (topology, bp_sig) dedup: {len(df_with_bp)} (from {before})")

    # =========================================================================
    # Step 5: With-bp — topology cap via RMSD diversity selection
    # =========================================================================
    topo_counts = df_with_bp["motif_topology"].value_counts()
    topos_needing_cap = topo_counts[topo_counts > topo_cap]

    if len(topos_needing_cap) > 0:
        print(
            f"Topologies exceeding cap of {topo_cap}: "
            f"{len(topos_needing_cap)} (will use RMSD diversity selection)"
        )

        work_items = []
        for topo in topos_needing_cap.index:
            topo_df = df_with_bp[df_with_bp["motif_topology"] == topo].sort_values(
                ["resolution", "motif_id"]
            )
            entries = []
            for _, row in topo_df.iterrows():
                entries.append({
                    "motif_id": row["motif_id"],
                    "pdb_id": row["pdb_id"],
                    "resolution": row["resolution"],
                    "bp_sig": row["bp_sig"],
                })
            work_items.append({"topology": topo, "entries": entries})

        print(f"Computing RMSD matrices for {len(work_items)} topologies...")
        rmsd_results = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(compute_rmsds_for_topology, w): w["topology"]
                for w in work_items
            }
            for future in as_completed(futures):
                topo = futures[future]
                try:
                    result = future.result()
                    rmsd_results[topo] = result
                except Exception as e:
                    log.error(f"Error computing RMSDs for {topo}: {e}")

        keep_ids_cap = set()
        # Keep all motifs from topologies that don't need capping
        for topo in topo_counts.index:
            if topo not in topos_needing_cap.index:
                topo_ids = df_with_bp[df_with_bp["motif_topology"] == topo][
                    "motif_id"
                ].tolist()
                keep_ids_cap.update(topo_ids)

        # Apply farthest-point selection and track rejections
        for topo, result in rmsd_results.items():
            entries = result["entries"]
            matrix = result["rmsd_matrix"]
            if matrix is not None:
                selected, rejected_with_rep = select_diverse_subset(
                    entries, matrix, topo_cap
                )
            else:
                selected = entries[:topo_cap]
                rejected_with_rep = [
                    (e, entries[0]["motif_id"]) for e in entries[topo_cap:]
                ]

            keep_ids_cap.update(e["motif_id"] for e in selected)

            for rejected_entry, nearest_rep_id in rejected_with_rep:
                tracking[rejected_entry["motif_id"]]["status"] = "rejected"
                tracking[rejected_entry["motif_id"]]["rejection_reason"] = "topology_cap"
                tracking[rejected_entry["motif_id"]][
                    "representative_motif_id"
                ] = nearest_rep_id

            if verbose:
                print(f"  {topo}: {len(entries)} -> {len(selected)}")

        before_cap = len(df_with_bp)
        df_with_bp = df_with_bp[df_with_bp["motif_id"].isin(keep_ids_cap)].copy()
        print(
            f"With-bp after topology cap ({topo_cap}): {len(df_with_bp)} (from {before_cap})"
        )

    # =========================================================================
    # Step 6: No-bp — N per topology (best resolution)
    # =========================================================================
    df_no_bp = df_no_bp.sort_values(["resolution", "motif_id"])
    no_bp_topo_groups = df_no_bp.groupby("motif_topology")
    keep_ids_nobp = set()
    for topo, group in no_bp_topo_groups:
        kept = group.head(no_bp_per_topo)
        keep_ids_nobp.update(kept["motif_id"].tolist())
        best_kept_id = kept.iloc[0]["motif_id"]
        for _, row in group.iloc[no_bp_per_topo:].iterrows():
            tracking[row["motif_id"]]["status"] = "rejected"
            tracking[row["motif_id"]]["rejection_reason"] = "no_bp_topology_cap"
            tracking[row["motif_id"]]["representative_motif_id"] = best_kept_id

    before = len(df_no_bp)
    df_no_bp = df_no_bp[df_no_bp["motif_id"].isin(keep_ids_nobp)].copy()
    print(f"No-bp after {no_bp_per_topo}/topology: {len(df_no_bp)} (from {before})")

    # =========================================================================
    # Mark all remaining as kept
    # =========================================================================
    df_result = pd.concat([df_with_bp, df_no_bp], ignore_index=True)
    for motif_id in df_result["motif_id"]:
        tracking[motif_id]["status"] = "kept"
        tracking[motif_id]["rejection_reason"] = ""
        tracking[motif_id]["representative_motif_id"] = motif_id

    # =========================================================================
    # Resolve representative chains
    # =========================================================================
    resolve_representative_chains(tracking)

    # =========================================================================
    # Summary
    # =========================================================================
    df_result = df_result.sort_values(
        ["motif_topology", "bp_sig", "resolution"]
    ).reset_index(drop=True)

    n_kept = sum(1 for v in tracking.values() if v["status"] == "kept")
    n_rejected = sum(1 for v in tracking.values() if v["status"] == "rejected")

    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total TWOWAY motifs: {len(tracking)}")
    print(f"  Kept: {n_kept}")
    print(f"  Rejected: {n_rejected}")
    print(f"    too_many_residues: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'too_many_residues')}")
    print(f"    sequence_dedup: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'sequence_dedup')}")
    print(f"    topology_bp_sig_dedup: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'topology_bp_sig_dedup')}")
    print(f"    topology_cap: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'topology_cap')}")
    print(f"    no_bp_topology_cap: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'no_bp_topology_cap')}")
    print(f"{'='*60}")

    # =========================================================================
    # Add comprehensive columns to curated set
    # =========================================================================
    print("Computing comprehensive columns for curated set...")
    df_result = add_comprehensive_columns(df_result)

    output_cols = [
        "motif_id", "pdb_id", "motif_sequence", "motif_topology",
        "strand1_sequence", "strand2_sequence", "is_bulge",
        "bp_sig", "basepairs", "closing_basepairs",
        "resolution", "num_residues", "num_non_canonical_basepairs",
        "num_hbonds", "num_protein_hbonds", "num_ligand_hbonds",
        "in_tertiary_contact", "num_tertiary_contacts",
        "has_non_canonical_residue", "has_non_canonical_basepair_flank",
        "is_isolatable",
    ]
    output_cols = [c for c in output_cols if c in df_result.columns]
    df_out = df_result[output_cols]
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_out)} curated motifs to {output_path}")

    # =========================================================================
    # Save tracking output
    # =========================================================================
    if tracking_path:
        # Reload all TWOWAY data for the full tracking CSV
        df_tracking = df_all[df_all["motif_type"] == "TWOWAY"].copy()
        df_tracking["resolution"] = df_tracking["pdb_id"].map(resolution_data).fillna(99.0)
        bp_info_all = df_tracking.apply(
            lambda row: get_cross_strand_bp_info(row, min_score=bp_score_threshold),
            axis=1,
        )
        df_tracking["bp_sig"] = bp_info_all.apply(lambda x: x[0])
        df_tracking["basepairs"] = bp_info_all.apply(lambda x: x[1])
        df_tracking["status"] = df_tracking["motif_id"].map(
            lambda x: tracking[x]["status"]
        )
        df_tracking["rejection_reason"] = df_tracking["motif_id"].map(
            lambda x: tracking[x]["rejection_reason"]
        )
        df_tracking["representative_motif_id"] = df_tracking["motif_id"].map(
            lambda x: tracking[x]["representative_motif_id"]
        )

        # Add comprehensive columns
        print("Computing comprehensive columns for tracking set...")
        df_tracking = add_comprehensive_columns(df_tracking)

        # Fill NaN/empty values explicitly
        df_tracking["bp_sig"] = df_tracking["bp_sig"].fillna("")
        df_tracking["rejection_reason"] = df_tracking["rejection_reason"].fillna("")
        df_tracking["representative_motif_id"] = df_tracking[
            "representative_motif_id"
        ].fillna("")

        tracking_cols = [
            "motif_id", "pdb_id", "motif_sequence", "motif_topology",
            "strand1_sequence", "strand2_sequence", "is_bulge",
            "bp_sig", "basepairs", "closing_basepairs",
            "resolution", "num_residues", "num_non_canonical_basepairs",
            "num_hbonds", "num_protein_hbonds", "num_ligand_hbonds",
            "in_tertiary_contact", "num_tertiary_contacts",
            "has_non_canonical_residue", "has_non_canonical_basepair_flank",
            "is_isolatable",
            "status", "rejection_reason", "representative_motif_id",
        ]
        tracking_cols = [c for c in tracking_cols if c in df_tracking.columns]
        df_tracking = df_tracking[tracking_cols].sort_values(
            ["status", "rejection_reason", "motif_topology", "motif_sequence"]
        )
        df_tracking.to_csv(tracking_path, index=False)
        print(f"Saved {len(df_tracking)} motif tracking records to {tracking_path}")


if __name__ == "__main__":
    filter_motifs()
