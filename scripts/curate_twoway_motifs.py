"""
Curate TWOWAY motifs by selecting high-quality, structurally diverse representatives.

This script reduces the filtered/validated TWOWAY motifs to a smaller set of
high-quality representatives through:
1. Adding resolution data and selecting best resolution per duplicate sequence
2. RMSD-based structural clustering to remove redundant structures
3. Multi-criteria filtering (cWW flanking, no modified nucleotides, ViennaRNA accuracy)
4. Basepair type diversity - motifs with different internal basepair patterns are kept

Usage:
    python scripts/curate_twoway_motifs.py -i validated_twoway.json -o curated_twoway.json

    # With custom thresholds
    python scripts/curate_twoway_motifs.py -i validated_twoway.json -o curated_twoway.json \
        --min-accuracy 0.9 --rmsd-threshold 1.0 --max-per-core 5

    # Skip RMSD clustering (faster, for testing)
    python scripts/curate_twoway_motifs.py -i validated_twoway.json -o curated_twoway.json --skip-rmsd
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_cached_motifs, Motif
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.tranforms import superimpose_structures, rmsd

log = get_logger("curate_twoway")


def load_basepair_data(pdb_id: str, motif_name: str) -> Tuple[int, str, List]:
    """
    Load basepair information for a motif from the motifs dataframe.

    Args:
        pdb_id: PDB ID
        motif_name: Full motif name

    Returns:
        Tuple of (num_non_canonical_bps, bp_signature, bp_details)
        - num_non_canonical_bps: count of internal non-canonical basepairs
        - bp_signature: sorted string of LW types (e.g., "cSH,cWW,tWW")
        - bp_details: list of [res1_type, res2_type, lw_type, score] for each bp
    """
    motifs_path = os.path.join(DATA_PATH, "dataframes", "motifs", f"{pdb_id}.json")
    if not os.path.exists(motifs_path):
        return (0, "", [])

    try:
        df = pd.read_json(motifs_path)
        match = df[df["motif_id"] == motif_name]
        if len(match) == 0:
            return (0, "", [])

        row = match.iloc[0]
        num_bps = row.get("num_non_canonical_basepairs", 0)
        non_canon_bps = row.get("non_canonical_bps", [])

        if not non_canon_bps or num_bps == 0:
            return (0, "", [])

        # Extract LW types and create signature
        lw_types = []
        bp_details = []
        for bp in non_canon_bps:
            # bp format: [res1_str, res2_str, lw_type, hbond_score]
            if len(bp) >= 3:
                lw_type = bp[2]
                lw_types.append(lw_type)
                # Extract residue types from res strings like "A-A-123-"
                res1_type = bp[0].split("-")[1] if "-" in bp[0] else bp[0]
                res2_type = bp[1].split("-")[1] if "-" in bp[1] else bp[1]
                score = bp[3] if len(bp) > 3 else 0.0
                bp_details.append([res1_type, res2_type, lw_type, score])

        # Create sorted signature for comparison
        bp_signature = ",".join(sorted(lw_types))

        return (num_bps, bp_signature, bp_details)

    except Exception as e:
        log.debug(f"Error loading basepair data for {motif_name}: {e}")
        return (0, "", [])


def load_resolution_data() -> Dict[str, float]:
    """
    Load resolution data from rna_structures.csv.

    Returns:
        Dict mapping pdb_id -> resolution (float)
    """
    resolution_path = os.path.join(DATA_PATH, "csvs", "rna_structures.csv")
    if not os.path.exists(resolution_path):
        log.warning(f"Resolution file not found: {resolution_path}")
        return {}

    df = pd.read_csv(resolution_path)
    return dict(zip(df["pdb_id"], df["resolution"]))




def get_motif_by_name(pdb_id: str, motif_name: str) -> Optional[Motif]:
    """
    Get a specific motif by name from cached motifs.

    Args:
        pdb_id: PDB ID
        motif_name: Full motif name

    Returns:
        Motif object or None if not found
    """
    try:
        motifs = get_cached_motifs(pdb_id)
        for m in motifs:
            if m.name == motif_name:
                return m
    except FileNotFoundError:
        log.warning(f"Motifs file not found for {pdb_id}")
    except Exception as e:
        log.warning(f"Error loading motifs for {pdb_id}: {e}")
    return None


def compute_pairwise_rmsd(
    motifs: List[Tuple[str, str, Motif]]
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise RMSD between motifs using C1' atoms.

    Args:
        motifs: List of (motif_name, pdb_id, Motif) tuples

    Returns:
        Dict mapping (name1, name2) -> RMSD value
    """
    rmsd_matrix = {}

    for i, (name1, pdb1, m1) in enumerate(motifs):
        coords1 = m1.get_c1prime_coords()
        if len(coords1) == 0:
            continue

        for j, (name2, pdb2, m2) in enumerate(motifs[i + 1 :], i + 1):
            coords2 = m2.get_c1prime_coords()
            if len(coords2) == 0 or len(coords1) != len(coords2):
                rmsd_matrix[(name1, name2)] = 999.0
                continue

            try:
                aligned_coords = superimpose_structures(coords2, coords1)
                rmsd_val = rmsd(coords1, aligned_coords)
                rmsd_matrix[(name1, name2)] = rmsd_val
            except Exception as e:
                log.debug(f"RMSD computation failed for {name1} vs {name2}: {e}")
                rmsd_matrix[(name1, name2)] = 999.0

    return rmsd_matrix


def cluster_by_rmsd(
    motifs: List[Tuple[str, str, float, Motif]], rmsd_threshold: float
) -> List[List[Tuple[str, str, float]]]:
    """
    Cluster motifs by RMSD using single-linkage clustering.

    Args:
        motifs: List of (motif_name, pdb_id, resolution, Motif) tuples
        rmsd_threshold: Maximum RMSD to consider structures as similar

    Returns:
        List of clusters, each cluster is a list of (motif_name, pdb_id, resolution)
    """
    if len(motifs) <= 1:
        return [[(name, pdb, res) for name, pdb, res, _ in motifs]]

    # Compute pairwise RMSD
    motif_data = [(name, pdb, m) for name, pdb, res, m in motifs]
    rmsd_matrix = compute_pairwise_rmsd(motif_data)

    # Build adjacency list for connected components (single-linkage)
    names = [name for name, _, _, _ in motifs]
    name_to_data = {name: (pdb, res) for name, pdb, res, _ in motifs}
    adjacency = defaultdict(set)

    for (name1, name2), rmsd_val in rmsd_matrix.items():
        if rmsd_val < rmsd_threshold:
            adjacency[name1].add(name2)
            adjacency[name2].add(name1)

    # Find connected components
    visited = set()
    clusters = []

    for name in names:
        if name in visited:
            continue

        # BFS to find all connected motifs
        cluster = []
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            pdb, res = name_to_data[current]
            cluster.append((current, pdb, res))
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        clusters.append(cluster)

    return clusters


def select_cluster_representative(
    cluster: List[Tuple[str, str, float]]
) -> Tuple[str, str, float]:
    """
    Select the best representative from a cluster (lowest resolution).

    Args:
        cluster: List of (motif_name, pdb_id, resolution) tuples

    Returns:
        Best representative tuple
    """
    # Sort by resolution (lowest first), then by name for determinism
    sorted_cluster = sorted(cluster, key=lambda x: (x[2], x[0]))
    return sorted_cluster[0]


def has_both_cww_flanking(row: pd.Series) -> bool:
    """
    Check if motif has cWW basepairs on both flanks.

    Uses expected_structure to infer flanking - if the expected structure
    has proper basepair notation at both ends, it likely has cWW flanking.
    """
    # Check the expected_structure for flanking basepair notation
    struct = row.get("expected_structure", "")
    if not struct:
        return False

    # Structure like "((.((&)).)" should have "(" at start and ")" at end
    # indicating flanking basepairs on both sides
    struct = struct.replace("&", "")
    if len(struct) < 4:
        return False

    # Check for basepair at 5' end (first char is '(')
    has_5p = struct[0] == "("
    # Check for basepair at 3' end (last char is ')')
    has_3p = struct[-1] == ")"

    return has_5p and has_3p


def has_modified_nucleotides(sequence: str) -> bool:
    """
    Check if sequence contains modified nucleotides (X).
    """
    return "X" in sequence


def parse_strand_sizes(sequence: str) -> Tuple[int, int]:
    """
    Parse strand sizes from sequence.

    Args:
        sequence: Sequence like "GAAAU-AUCUC"

    Returns:
        Tuple of (strand1_size, strand2_size)
    """
    parts = sequence.split("-")
    if len(parts) != 2:
        return (0, 0)
    return (len(parts[0]), len(parts[1]))


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON from validate_twoway_structures.py or filter_twoway_motifs.py",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    help="Output JSON path",
)
@click.option(
    "--require-both-cww/--no-require-both-cww",
    default=True,
    help="Require both flanking basepairs to be cWW (default: True)",
)
@click.option(
    "--exclude-modified/--allow-modified",
    default=True,
    help="Exclude sequences with modified nucleotides (default: True)",
)
@click.option(
    "--min-accuracy",
    type=float,
    default=0.9,
    help="Minimum ViennaRNA prediction accuracy (default: 0.9)",
)
@click.option(
    "--max-strand-size",
    type=int,
    default=8,
    help="Maximum nucleotides per strand (default: 8)",
)
@click.option(
    "--max-total-size",
    type=int,
    default=13,
    help="Maximum total nucleotides across both strands (default: 13)",
)
@click.option(
    "--rmsd-threshold",
    type=float,
    default=1.0,
    help="RMSD threshold for clustering similar structures (default: 1.0 Å)",
)
@click.option(
    "--max-per-core",
    type=int,
    default=5,
    help="Maximum motifs to keep per core sequence (default: 5)",
)
@click.option(
    "--skip-rmsd",
    is_flag=True,
    help="Skip RMSD clustering (faster, for testing)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed progress",
)
def curate_motifs(
    input_path,
    output_path,
    require_both_cww,
    exclude_modified,
    min_accuracy,
    max_strand_size,
    max_total_size,
    rmsd_threshold,
    max_per_core,
    skip_rmsd,
    verbose,
):
    """Curate TWOWAY motifs to select high-quality, diverse representatives."""

    # Load input data
    log.info(f"Reading {input_path}")
    df = pd.read_json(input_path)
    initial_count = len(df)
    log.info(f"Loaded {initial_count} motifs")

    # Track filtering stats
    stats = {"initial": initial_count}

    # =========================================================================
    # Phase 1: Add resolution data and select best per extended_sequence
    # =========================================================================
    log.info("Phase 1: Adding resolution data")

    resolution_data = load_resolution_data()
    df["resolution"] = df["pdb_id"].map(resolution_data)
    # Assign penalty resolution for missing data
    df["resolution"] = df["resolution"].fillna(99.0)

    # Sort by resolution (best first) within each extended_sequence group
    df = df.sort_values(["extended_sequence", "resolution", "motif_name"])

    # Keep best resolution per extended_sequence
    before = len(df)
    df = df.drop_duplicates(subset=["extended_sequence"], keep="first")
    stats["after_resolution_dedup"] = len(df)
    log.info(f"Best resolution per extended_sequence: {before} -> {len(df)}")

    # =========================================================================
    # Phase 2: Multi-criteria filtering
    # =========================================================================
    log.info("Phase 2: Applying multi-criteria filters")

    # The 'sequence' column already contains the core motif sequence (internal loop
    # nucleotides without flanking basepair residues). For example:
    # - extended_sequence: "GAAAU-AUCUC" (includes flanking G-C and U-C basepairs)
    # - sequence: "AAA-UCU" (core internal loop sequence)
    df["core_sequence"] = df["sequence"]

    # Add strand size columns
    strand_sizes = df["extended_sequence"].apply(parse_strand_sizes)
    df["strand1_size"] = strand_sizes.apply(lambda x: x[0])
    df["strand2_size"] = strand_sizes.apply(lambda x: x[1])

    # Filter: Both cWW flanking
    if require_both_cww:
        before = len(df)
        df["has_both_cww"] = df.apply(has_both_cww_flanking, axis=1)
        df = df[df["has_both_cww"]]
        stats["after_cww_filter"] = len(df)
        log.info(f"Both cWW flanking: {before} -> {len(df)}")

    # Filter: No modified nucleotides
    if exclude_modified:
        before = len(df)
        df = df[~df["extended_sequence"].apply(has_modified_nucleotides)]
        stats["after_modified_filter"] = len(df)
        log.info(f"Exclude modified nucleotides: {before} -> {len(df)}")

    # Filter: ViennaRNA accuracy
    if "accuracy" in df.columns and min_accuracy is not None:
        before = len(df)
        df = df[df["accuracy"] >= min_accuracy]
        stats["after_accuracy_filter"] = len(df)
        log.info(f"Min accuracy >= {min_accuracy}: {before} -> {len(df)}")

    # Add total size column
    df["total_size"] = df["strand1_size"] + df["strand2_size"]

    # Filter: Max strand size
    if max_strand_size is not None:
        before = len(df)
        df = df[
            (df["strand1_size"] <= max_strand_size)
            & (df["strand2_size"] <= max_strand_size)
        ]
        stats["after_strand_size_filter"] = len(df)
        log.info(f"Max strand size <= {max_strand_size}: {before} -> {len(df)}")

    # Filter: Max total size
    if max_total_size is not None:
        before = len(df)
        df = df[df["total_size"] <= max_total_size]
        stats["after_total_size_filter"] = len(df)
        log.info(f"Max total size <= {max_total_size}: {before} -> {len(df)}")

    # =========================================================================
    # Phase 3: Load basepair data and add to dataframe
    # =========================================================================
    log.info("Phase 3: Loading basepair interaction data")

    bp_data = []
    for _, row in df.iterrows():
        num_bps, bp_sig, bp_details = load_basepair_data(row["pdb_id"], row["motif_name"])
        bp_data.append({
            "num_non_canonical_bps": num_bps,
            "bp_signature": bp_sig,
            "bp_types": bp_sig,  # Human-readable column
        })

    bp_df = pd.DataFrame(bp_data)
    df = df.reset_index(drop=True)
    df["num_non_canonical_bps"] = bp_df["num_non_canonical_bps"]
    df["bp_signature"] = bp_df["bp_signature"]
    df["bp_types"] = bp_df["bp_types"]

    log.info(f"Loaded basepair data for {len(df)} motifs")
    log.info(f"Unique basepair signatures: {df['bp_signature'].nunique()}")

    # =========================================================================
    # Phase 4: RMSD-based structural clustering (within same bp_signature)
    # =========================================================================
    if not skip_rmsd:
        log.info("Phase 4: RMSD-based structural clustering")
        log.info("  (Motifs with different basepair types are kept separate)")

        # Group by BOTH core sequence AND basepair signature
        # This ensures motifs with different interaction patterns are not clustered together
        df["cluster_key"] = df["core_sequence"] + "|" + df["bp_signature"]
        cluster_groups = df.groupby("cluster_key")
        clustered_rows = []
        cluster_id_counter = 0

        total_groups = len(cluster_groups)
        for idx, (cluster_key, group) in enumerate(cluster_groups):
            if verbose and idx % 50 == 0:
                log.info(f"Processing group {idx + 1}/{total_groups}: {cluster_key}")

            if len(group) == 1:
                # Single motif in group, no clustering needed
                row = group.iloc[0].copy()
                row["cluster_id"] = cluster_id_counter
                row["cluster_size"] = 1
                row["cluster_rmsd"] = 0.0
                clustered_rows.append(row)
                cluster_id_counter += 1
                continue

            # Load motif structures for RMSD calculation
            motifs_with_data = []
            for _, row in group.iterrows():
                motif = get_motif_by_name(row["pdb_id"], row["motif_name"])
                if motif is not None:
                    motifs_with_data.append(
                        (row["motif_name"], row["pdb_id"], row["resolution"], motif)
                    )

            if len(motifs_with_data) == 0:
                log.warning(f"No motif structures loaded for {cluster_key}")
                continue

            if len(motifs_with_data) == 1:
                # Only one motif could be loaded
                name, pdb, res, _ = motifs_with_data[0]
                row = group[group["motif_name"] == name].iloc[0].copy()
                row["cluster_id"] = cluster_id_counter
                row["cluster_size"] = 1
                row["cluster_rmsd"] = 0.0
                clustered_rows.append(row)
                cluster_id_counter += 1
                continue

            # Cluster by RMSD (only within same bp_signature group)
            clusters = cluster_by_rmsd(motifs_with_data, rmsd_threshold)

            for cluster in clusters:
                # Select best representative (lowest resolution)
                repr_name, repr_pdb, repr_res = select_cluster_representative(cluster)

                # Get the row for the representative
                row = group[group["motif_name"] == repr_name].iloc[0].copy()
                row["cluster_id"] = cluster_id_counter
                row["cluster_size"] = len(cluster)
                row["cluster_rmsd"] = 0.0  # Representative has 0 RMSD to itself
                clustered_rows.append(row)
                cluster_id_counter += 1

        df = pd.DataFrame(clustered_rows)
        stats["after_rmsd_clustering"] = len(df)
        log.info(f"After RMSD clustering: {stats.get('after_size_filter', stats.get('after_accuracy_filter', len(df)))} -> {len(df)}")
    else:
        df["cluster_id"] = range(len(df))
        df["cluster_size"] = 1
        df["cluster_rmsd"] = 0.0

    # =========================================================================
    # Phase 5: Limit per core sequence
    # =========================================================================
    if max_per_core is not None:
        log.info(f"Phase 5: Limiting to {max_per_core} per core sequence")
        before = len(df)

        # Sort by resolution within each core sequence, then take top N
        df = df.sort_values(["core_sequence", "resolution", "motif_name"])
        df = df.groupby("core_sequence").head(max_per_core)

        stats["after_max_per_core"] = len(df)
        log.info(f"Max {max_per_core} per core: {before} -> {len(df)}")

    # =========================================================================
    # Prepare output
    # =========================================================================
    log.info("Preparing output")

    # Select and order output columns
    output_cols = [
        "motif_name",
        "pdb_id",
        "sequence",
        "extended_sequence",
        "core_sequence",
        "expected_structure",
        "predicted_structure",
        "accuracy",
        "resolution",
        "num_non_canonical_bps",
        "bp_types",
        "cluster_id",
        "cluster_size",
        "strand1_size",
        "strand2_size",
    ]

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()

    # Sort output by core_sequence, then resolution
    df_out = df_out.sort_values(["core_sequence", "resolution", "motif_name"])
    df_out = df_out.reset_index(drop=True)

    # Save output
    df_out.to_json(output_path, orient="records", indent=2)
    log.info(f"Saved {len(df_out)} curated motifs to {output_path}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("CURATION SUMMARY")
    print("=" * 70)
    print(f"  Input:  {stats['initial']} motifs")

    if "after_resolution_dedup" in stats:
        print(f"  After resolution dedup: {stats['after_resolution_dedup']}")
    if "after_cww_filter" in stats:
        print(f"  After cWW filter: {stats['after_cww_filter']}")
    if "after_modified_filter" in stats:
        print(f"  After modified filter: {stats['after_modified_filter']}")
    if "after_accuracy_filter" in stats:
        print(f"  After accuracy filter: {stats['after_accuracy_filter']}")
    if "after_strand_size_filter" in stats:
        print(f"  After strand size filter: {stats['after_strand_size_filter']}")
    if "after_total_size_filter" in stats:
        print(f"  After total size filter: {stats['after_total_size_filter']}")
    if "after_rmsd_clustering" in stats:
        print(f"  After RMSD clustering: {stats['after_rmsd_clustering']}")
    if "after_max_per_core" in stats:
        print(f"  After max-per-core limit: {stats['after_max_per_core']}")

    print(f"\n  Final output: {len(df_out)} motifs")
    reduction = (1 - len(df_out) / stats["initial"]) * 100
    print(f"  Reduction: {stats['initial'] - len(df_out)} ({reduction:.1f}%)")

    # Core sequence distribution
    print(f"\n  Unique core sequences: {df_out['core_sequence'].nunique()}")

    # Size distribution
    print("\n  Size distribution (strand1-strand2):")
    df_out["size_str"] = (
        df_out["strand1_size"].astype(str) + "-" + df_out["strand2_size"].astype(str)
    )
    size_counts = df_out["size_str"].value_counts().head(10)
    for size, count in size_counts.items():
        print(f"    {size}: {count}")

    # Resolution distribution
    if "resolution" in df_out.columns:
        print("\n  Resolution distribution:")
        print(f"    Min: {df_out['resolution'].min():.2f} Å")
        print(f"    Max: {df_out['resolution'].max():.2f} Å")
        print(f"    Mean: {df_out['resolution'].mean():.2f} Å")
        print(f"    Median: {df_out['resolution'].median():.2f} Å")

    # Accuracy distribution
    if "accuracy" in df_out.columns:
        print("\n  Accuracy distribution:")
        print(f"    Min: {df_out['accuracy'].min():.2%}")
        print(f"    Max: {df_out['accuracy'].max():.2%}")
        print(f"    Mean: {df_out['accuracy'].mean():.2%}")

    # Basepair type distribution
    if "bp_types" in df_out.columns:
        print("\n  Basepair types (top 10):")
        bp_counts = df_out["bp_types"].value_counts().head(10)
        for bp_type, count in bp_counts.items():
            label = bp_type if bp_type else "(none)"
            print(f"    {label}: {count}")

        # Count motifs with non-canonical basepairs
        has_bp = (df_out["num_non_canonical_bps"] > 0).sum()
        print(f"\n  Motifs with internal basepairs: {has_bp} ({has_bp/len(df_out)*100:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    curate_motifs()
