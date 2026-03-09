"""
Filter and deduplicate TWOWAY motifs to a curated representative set.

Reads from data/summaries/motifs/non_redundant_motifs_summary.json and produces:
1. A curated JSON of ~456 unique representative motifs
2. A tracking JSON of ALL 16,806 motifs with status, rejection reason, and representative

Pipeline:
1. Max residue filter (default: 15)
2. Sequence dedup (keep best resolution per unique sequence)
3. Compute cross-strand basepair signature (score >= 1.0)
4. Split into with-basepair and no-basepair populations
5. With-bp: dedup by (topology, bp_sig), then cap per topology
   using farthest-point RMSD diversity selection
6. No-bp: keep N per topology (best resolution)

Usage:
    python scripts/filter_twoway_motifs.py -o curated_twoway.json
    python scripts/filter_twoway_motifs.py -o curated.json --tracking-output tracking.json
    python scripts/filter_twoway_motifs.py -o curated.json --topo-cap 20 --no-bp-per-topo 2
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import Random

import click
import numpy as np
import pandas as pd
import RNA

from rna_motif_library.basepair import get_cached_basepairs
from rna_motif_library.chain import Chains, get_cached_chains
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


WC_PAIRS = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]


def _random_helix(length, rng):
    """Generate a random Watson-Crick helix."""
    bp_combo = [rng.choice(WC_PAIRS) for _ in range(length)]
    strand1 = "".join(bp[0] for bp in bp_combo)
    strand2 = "".join(bp[1] for bp in reversed(bp_combo))
    return strand1, strand2


def get_internal_residues(motif_sequence):
    """
    Get internal (unpaired/dot) residues from a motif sequence.

    Strips the first and last residue from each strand (closing basepair
    residues) and returns the remaining internal nucleotides.
    """
    parts = motif_sequence.split("-")
    if len(parts) != 2:
        return ""
    s1, s2 = parts
    internal_s1 = s1[1:-1] if len(s1) > 2 else ""
    internal_s2 = s2[1:-1] if len(s2) > 2 else ""
    return internal_s1 + internal_s2


def has_ac_internal(motif_sequence):
    """Check if motif has at least one A or C in internal positions."""
    internal = get_internal_residues(motif_sequence)
    if not internal:
        return True  # No internal residues to check
    return any(n in "AC" for n in internal)


def count_ac_internal(motif_sequence):
    """Count A/C nucleotides in internal positions (for sorting priority)."""
    internal = get_internal_residues(motif_sequence)
    if not internal:
        return 0
    return sum(1 for n in internal if n in "AC")


def compute_vienna_accuracy(motif_sequence, motif_topology, helix_len=5, seed=42):
    """
    Compute ViennaRNA prediction accuracy for a TWOWAY motif.

    Embeds the motif in a hairpin construct and checks if ViennaRNA
    correctly predicts the flanking pairs and internal unpaired regions.

    Returns accuracy (0.0-1.0) for the motif region.
    """
    parts = motif_sequence.split("-")
    if len(parts) != 2:
        return 0.0
    s1_seq, s2_seq = parts
    if "X" in s1_seq or "X" in s2_seq:
        return 0.0

    topo_parts = motif_topology.split("-")
    t1, t2 = int(topo_parts[0]), int(topo_parts[1])

    # Expected structure: flanking pairs + internal dots
    struct_s1 = "(" + "." * t1 + "("
    struct_s2 = ")" + "." * t2 + ")"

    # Build hairpin construct: helix + motif_s1 + helix + GAAA + helix + motif_s2 + helix
    rng = Random(seed)
    h1_s1, h1_s2 = _random_helix(helix_len, rng)
    h2_s1, h2_s2 = _random_helix(helix_len, rng)

    full_seq = h1_s1 + s1_seq + h2_s1 + "GAAA" + h2_s2 + s2_seq + h1_s2
    h_open = "(" * helix_len
    h_close = ")" * helix_len
    full_expected = h_open + struct_s1 + h_open + "...." + h_close + struct_s2 + h_close

    # Fold
    fc = RNA.fold_compound(full_seq)
    predicted, _ = fc.mfe()

    # Compute accuracy for motif region only
    s1_start = helix_len
    s1_end = s1_start + len(s1_seq)
    s2_start = s1_end + helix_len + 4 + helix_len
    s2_end = s2_start + len(s2_seq)

    total = 0
    matches = 0
    for i in range(s1_start, s1_end):
        total += 1
        if full_expected[i] == predicted[i]:
            matches += 1
    for i in range(s2_start, s2_end):
        total += 1
        if full_expected[i] == predicted[i]:
            matches += 1

    return matches / total if total > 0 else 0.0


def derive_structure(motif_topology):
    """
    Derive dot-bracket structure for a TWOWAY motif from its topology.

    Flanking residues are paired, internal residues are unpaired.
    E.g. topology "1-0" -> "(.(&))", topology "2-3" -> "(..(&...))"
    """
    parts = motif_topology.split("-")
    if len(parts) != 2:
        return ""
    t1, t2 = int(parts[0]), int(parts[1])
    struct_s1 = "(" + "." * t1 + "("
    struct_s2 = ")" + "." * t2 + ")"
    return f"{struct_s1}&{struct_s2}"


def compute_extended_seq_struct(motif_id, pdb_id, motif_sequence, motif_topology):
    """
    Compute extended sequence and structure by adding the flanking helix
    basepair outside the motif.

    Returns (extended_sequence, extended_structure) or (motif_sequence, structure)
    if flanking data can't be loaded.
    """
    structure = derive_structure(motif_topology)
    try:
        motifs = get_cached_motifs(pdb_id)
        chains_list = get_cached_chains(pdb_id)
        chains = Chains(chains_list)
        basepairs = get_cached_basepairs(pdb_id)
    except Exception:
        return motif_sequence, structure

    # Find motif
    motif = None
    for m in motifs:
        if m.name == motif_id:
            motif = m
            break
    if motif is None or len(motif.strands) != 2:
        return motif_sequence, structure

    # Build basepair lookup
    bp_lookup = {}
    for bp in basepairs:
        k1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
        k2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
        bp_lookup[k1] = bp
        bp_lookup[k2] = bp

    # Get flanking residues in the chain
    s0, s1 = motif.strands[0], motif.strands[1]
    first_0 = chains.get_residue_by_str(s0[0].get_str())
    last_0 = chains.get_residue_by_str(s0[-1].get_str())
    first_1 = chains.get_residue_by_str(s1[0].get_str())
    last_1 = chains.get_residue_by_str(s1[-1].get_str())

    prev_0 = chains.get_previous_residue_in_chain(first_0) if first_0 else None
    next_0 = chains.get_next_residue_in_chain(last_0) if last_0 else None
    prev_1 = chains.get_previous_residue_in_chain(first_1) if first_1 else None
    next_1 = chains.get_next_residue_in_chain(last_1) if last_1 else None

    # Check for flanking basepairs (antiparallel)
    def _check_bp(r1, r2):
        if r1 is None or r2 is None:
            return None
        return bp_lookup.get(f"{r1.get_str()}-{r2.get_str()}")

    bp_5p = _check_bp(prev_0, next_1)
    bp_3p = _check_bp(next_0, prev_1)

    seq_parts = motif_sequence.split("-")
    struct_parts = structure.split("&")
    ext_s0_seq, ext_s1_seq = seq_parts[0], seq_parts[1]
    ext_s0_ss, ext_s1_ss = struct_parts[0], struct_parts[1]

    def _nuc(res):
        return res.res_id if res.res_id in "AGCU" else "X"

    if bp_5p is not None:
        ext_s0_seq = _nuc(prev_0) + ext_s0_seq
        ext_s1_seq = ext_s1_seq + _nuc(next_1)
        ext_s0_ss = "(" + ext_s0_ss
        ext_s1_ss = ext_s1_ss + ")"

    if bp_3p is not None:
        ext_s0_seq = ext_s0_seq + _nuc(next_0)
        ext_s1_seq = _nuc(prev_1) + ext_s1_seq
        ext_s0_ss = ext_s0_ss + "("
        ext_s1_ss = ")" + ext_s1_ss

    return f"{ext_s0_seq}-{ext_s1_seq}", f"{ext_s0_ss}&{ext_s1_ss}"


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
        return "", []
    seq_parts = row["motif_sequence"].split("-")
    if len(seq_parts) != 2:
        return "", []
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
    return ",".join(sorted(lw_types)), bp_details


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
    help="Output JSON path for curated motifs",
)
@click.option(
    "--tracking-output", "tracking_path", default=None,
    help="Output JSON path for full tracking of all motifs (optional)",
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
    df_tw["structure"] = df_tw["motif_topology"].apply(derive_structure)

    # Initialize tracking for ALL motifs
    tracking = {}
    for _, row in df_tw.iterrows():
        tracking[row["motif_id"]] = {
            "status": "pending",
            "rejection_reason": "",
            "representative_motif_id": "",
        }

    # =========================================================================
    # Step 1a: Isolatable filter
    # =========================================================================
    not_isolatable = df_tw["is_isolatable"] != 1
    for motif_id in df_tw.loc[not_isolatable, "motif_id"]:
        tracking[motif_id]["status"] = "rejected"
        tracking[motif_id]["rejection_reason"] = "not_isolatable"

    df_tw = df_tw[~not_isolatable].copy()
    print(f"After is_isolatable filter: {len(df_tw)} (removed {not_isolatable.sum()})")

    # =========================================================================
    # Step 1b: Max residue filter
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
    # Step 2.5: Require at least one A/C in internal (dot) positions
    # =========================================================================
    no_ac = ~df_tw["motif_sequence"].apply(has_ac_internal)
    for motif_id in df_tw.loc[no_ac, "motif_id"]:
        tracking[motif_id]["status"] = "rejected"
        tracking[motif_id]["rejection_reason"] = "no_ac_internal"

    df_tw = df_tw[~no_ac].copy()
    print(f"After A/C internal filter: {len(df_tw)} (removed {no_ac.sum()})")

    # =========================================================================
    # Step 2.6: ViennaRNA secondary structure validation
    # =========================================================================
    print(f"Computing ViennaRNA accuracy for {len(df_tw)} motifs...")
    df_tw["accuracy"] = df_tw.apply(
        lambda row: compute_vienna_accuracy(
            row["motif_sequence"], row["motif_topology"]
        ),
        axis=1,
    )
    # Priority: small motifs (<=6 residues) with good structure get priority 0
    df_tw["_structure_priority"] = 1
    small_and_good = (df_tw["num_residues"] <= 6) & (df_tw["accuracy"] >= 0.9)
    df_tw.loc[small_and_good, "_structure_priority"] = 0
    # A/C count for sorting: more A/C in internal positions = better (negate for ascending sort)
    df_tw["_ac_count"] = -df_tw["motif_sequence"].apply(count_ac_internal)
    n_favored = small_and_good.sum()
    mean_acc = df_tw["accuracy"].mean()
    perfect = (df_tw["accuracy"] == 1.0).sum()
    print(
        f"  Mean accuracy: {mean_acc:.3f}, perfect: {perfect}/{len(df_tw)}, "
        f"small+good (<=6 res, >=0.9 acc): {n_favored}"
    )

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
    df_with_bp = df_with_bp.sort_values(["_structure_priority", "_ac_count", "resolution", "motif_id"])

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
    # Step 6: No-bp — N per topology, prioritizing A/C bulge residues
    # =========================================================================

    def _bulge_ac_priority(row):
        """Score bulge motifs: 0 if all internal residues are A/C, 1 otherwise."""
        topo_parts = row["motif_topology"].split("-")
        if "0" not in topo_parts:
            return 0
        seq_parts = row["motif_sequence"].split("-")
        if topo_parts[1] == "0":
            bulge_seq = seq_parts[0]
        else:
            bulge_seq = seq_parts[1]
        # Strip flanking residues to get internal bulge nucleotides
        internal = bulge_seq[1:-1] if len(bulge_seq) > 2 else bulge_seq
        return 0 if all(n in "AC" for n in internal) else 1

    df_no_bp["_ac_priority"] = df_no_bp.apply(_bulge_ac_priority, axis=1)
    df_no_bp = df_no_bp.sort_values(
        ["_structure_priority", "_ac_priority", "_ac_count", "resolution", "motif_id"]
    )
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
    df_no_bp = df_no_bp.drop(columns=["_ac_priority"])

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
    print(f"    not_isolatable: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'not_isolatable')}")
    print(f"    too_many_residues: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'too_many_residues')}")
    print(f"    sequence_dedup: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'sequence_dedup')}")
    print(f"    no_ac_internal: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'no_ac_internal')}")
    print(f"    topology_bp_sig_dedup: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'topology_bp_sig_dedup')}")
    print(f"    topology_cap: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'topology_cap')}")
    print(f"    no_bp_topology_cap: {sum(1 for v in tracking.values() if v['rejection_reason'] == 'no_bp_topology_cap')}")
    print(f"{'='*60}")

    # =========================================================================
    # Add extended sequence/structure (with second flanking pair from helix)
    # =========================================================================
    print("Computing extended sequences with flanking helix basepairs...")
    ext_data = df_result.apply(
        lambda row: compute_extended_seq_struct(
            row["motif_id"], row["pdb_id"],
            row["motif_sequence"], row["motif_topology"],
        ),
        axis=1,
    )
    df_result["sequence"] = ext_data.apply(lambda x: x[0])
    df_result["structure"] = ext_data.apply(lambda x: x[1])

    # =========================================================================
    # Add comprehensive columns to curated set
    # =========================================================================
    print("Computing comprehensive columns for curated set...")
    df_result = add_comprehensive_columns(df_result)

    output_cols = [
        "motif_id", "pdb_id", "sequence", "structure",
        "motif_sequence", "motif_topology",
        "strand1_sequence", "strand2_sequence", "is_bulge",
        "bp_sig", "basepairs",
        "resolution", "accuracy", "num_residues", "num_non_canonical_basepairs",
        "num_hbonds", "num_protein_hbonds", "num_ligand_hbonds",
        "in_tertiary_contact", "num_tertiary_contacts",
        "has_non_canonical_residue", "has_non_canonical_basepair_flank",
        "is_isolatable",
    ]
    output_cols = [c for c in output_cols if c in df_result.columns]
    df_out = df_result[output_cols]
    df_out.to_json(output_path, orient="records", indent=2)
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
        df_tracking["structure"] = df_tracking["motif_topology"].apply(derive_structure)
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
            "motif_id", "pdb_id", "motif_sequence", "structure", "motif_topology",
            "strand1_sequence", "strand2_sequence", "is_bulge",
            "bp_sig", "basepairs",
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
        df_tracking.to_json(tracking_path, orient="records", indent=2)
        print(f"Saved {len(df_tracking)} motif tracking records to {tracking_path}")


if __name__ == "__main__":
    filter_motifs()
