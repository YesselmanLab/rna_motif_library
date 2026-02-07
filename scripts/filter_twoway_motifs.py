"""
Filter and deduplicate TWOWAY motifs from flanking basepair analysis.

This script filters the output from process_twoway_flanking_basepairs.py,
removing duplicate sequences and applying various filters.

Usage:
    # Basic deduplication
    python scripts/filter_twoway_motifs.py -i twoway_flanking_basepairs.csv -o filtered.csv

    # Filter by size (max 5 residues per strand)
    python scripts/filter_twoway_motifs.py -i twoway_flanking_basepairs.csv -o filtered.csv --max-size 5

    # Only motifs with both flanking basepairs
    python scripts/filter_twoway_motifs.py -i twoway_flanking_basepairs.csv -o filtered.csv --require-both-flanking

    # Combine filters
    python scripts/filter_twoway_motifs.py -i input.csv -o output.csv --max-size 4 --require-both-flanking --bp-type cWW
"""

import click
import pandas as pd

from rna_motif_library.logger import get_logger

log = get_logger("filter_twoway")


def parse_size(sequence: str) -> tuple:
    """
    Parse sequence to get strand sizes.

    Args:
        sequence: Sequence string like "AAAAGUC-GUCGCU"

    Returns:
        Tuple of (strand1_size, strand2_size, total_size)
    """
    parts = sequence.split("-")
    if len(parts) != 2:
        return (0, 0, 0)
    s1, s2 = len(parts[0]), len(parts[1])
    return (s1, s2, s1 + s2)


def get_size_string(sequence: str) -> str:
    """Get size string like '5-4' from sequence."""
    parts = sequence.split("-")
    if len(parts) != 2:
        return "0-0"
    return f"{len(parts[0])}-{len(parts[1])}"


def count_unpaired_nucleotides(sequence: str, structure: str) -> dict:
    """
    Count nucleotides in unpaired positions (dots in structure).

    Args:
        sequence: Sequence string like "AAAAGUC-GUCGCU"
        structure: Structure string like "(.....(&)....)"

    Returns:
        Dict with counts of A, C, G, U in unpaired positions
    """
    # Remove separators to align sequence and structure
    seq = sequence.replace("-", "")
    struct = structure.replace("&", "")

    if len(seq) != len(struct):
        return {"A": 0, "C": 0, "G": 0, "U": 0, "AC": 0, "total_unpaired": 0}

    counts = {"A": 0, "C": 0, "G": 0, "U": 0}
    for nuc, ss in zip(seq, struct):
        if ss == ".":
            if nuc in counts:
                counts[nuc] += 1

    counts["AC"] = counts["A"] + counts["C"]
    counts["total_unpaired"] = sum(counts[n] for n in "AGCU")
    return counts


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input CSV from process_twoway_flanking_basepairs.py",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    help="Output CSV path",
)
@click.option(
    "--max-size",
    type=int,
    default=None,
    help="Maximum residues per strand (e.g., 5 means each strand <= 5 residues)",
)
@click.option(
    "--min-size",
    type=int,
    default=None,
    help="Minimum residues per strand",
)
@click.option(
    "--max-total",
    type=int,
    default=None,
    help="Maximum total residues across both strands",
)
@click.option(
    "--require-both-flanking",
    is_flag=True,
    help="Only include motifs with both 5' and 3' flanking basepairs",
)
@click.option(
    "--require-5p",
    is_flag=True,
    help="Only include motifs with 5' flanking basepair",
)
@click.option(
    "--require-3p",
    is_flag=True,
    help="Only include motifs with 3' flanking basepair",
)
@click.option(
    "--bp-type",
    type=str,
    default=None,
    help="Filter by flanking basepair type (e.g., 'cWW' for canonical Watson-Crick)",
)
@click.option(
    "--exclude-modified",
    is_flag=True,
    help="Exclude sequences containing modified nucleotides (X)",
)
@click.option(
    "--min-unpaired-ac",
    type=int,
    default=1,
    help="Minimum A+C count in unpaired regions (default: 1, removes motifs with 0)",
)
@click.option(
    "--max-unpaired-ac",
    type=int,
    default=None,
    help="Maximum A+C count in unpaired regions",
)
@click.option(
    "--symmetric",
    is_flag=True,
    help="Only include symmetric motifs (same size on both strands)",
)
@click.option(
    "--keep-duplicates",
    is_flag=True,
    help="Keep duplicate sequences (default: remove duplicates)",
)
@click.option(
    "--dedupe-by",
    type=click.Choice(["sequence", "extended_sequence", "structure"]),
    default="extended_sequence",
    help="Field to use for deduplication (default: extended_sequence)",
)
@click.option(
    "--sort-by",
    type=click.Choice(["sequence", "size", "pdb_id", "count"]),
    default="sequence",
    help="Sort output by this field",
)
@click.option(
    "--add-counts",
    is_flag=True,
    help="Add a column showing how many times each sequence appears",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed filtering statistics",
)
def filter_motifs(
    input_path,
    output_path,
    max_size,
    min_size,
    max_total,
    require_both_flanking,
    require_5p,
    require_3p,
    bp_type,
    exclude_modified,
    min_unpaired_ac,
    max_unpaired_ac,
    symmetric,
    keep_duplicates,
    dedupe_by,
    sort_by,
    add_counts,
    verbose,
):
    """Filter and deduplicate TWOWAY motifs."""
    # Load input
    log.info(f"Reading {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    log.info(f"Loaded {initial_count} motifs")

    # Add size columns for filtering
    df["strand1_size"] = df["sequence"].apply(lambda x: parse_size(x)[0])
    df["strand2_size"] = df["sequence"].apply(lambda x: parse_size(x)[1])
    df["total_size"] = df["sequence"].apply(lambda x: parse_size(x)[2])
    df["size_str"] = df["sequence"].apply(get_size_string)

    # Add unpaired nucleotide counts
    unpaired_counts = df.apply(
        lambda row: count_unpaired_nucleotides(row["sequence"], row["structure"]),
        axis=1,
    )
    df["unpaired_A"] = unpaired_counts.apply(lambda x: x["A"])
    df["unpaired_C"] = unpaired_counts.apply(lambda x: x["C"])
    df["unpaired_AC"] = unpaired_counts.apply(lambda x: x["AC"])
    df["total_unpaired"] = unpaired_counts.apply(lambda x: x["total_unpaired"])

    # Apply filters
    filters_applied = []

    # Size filters
    if max_size is not None:
        before = len(df)
        df = df[(df["strand1_size"] <= max_size) & (df["strand2_size"] <= max_size)]
        filters_applied.append(f"max_size={max_size}: {before} -> {len(df)}")

    if min_size is not None:
        before = len(df)
        df = df[(df["strand1_size"] >= min_size) & (df["strand2_size"] >= min_size)]
        filters_applied.append(f"min_size={min_size}: {before} -> {len(df)}")

    if max_total is not None:
        before = len(df)
        df = df[df["total_size"] <= max_total]
        filters_applied.append(f"max_total={max_total}: {before} -> {len(df)}")

    if symmetric:
        before = len(df)
        df = df[df["strand1_size"] == df["strand2_size"]]
        filters_applied.append(f"symmetric: {before} -> {len(df)}")

    # Flanking basepair filters
    if require_both_flanking:
        before = len(df)
        df = df[(df["flanking_bp_type_5p"] != "") & (df["flanking_bp_type_3p"] != "")]
        filters_applied.append(f"require_both_flanking: {before} -> {len(df)}")

    if require_5p:
        before = len(df)
        df = df[df["flanking_bp_type_5p"] != ""]
        filters_applied.append(f"require_5p: {before} -> {len(df)}")

    if require_3p:
        before = len(df)
        df = df[df["flanking_bp_type_3p"] != ""]
        filters_applied.append(f"require_3p: {before} -> {len(df)}")

    if bp_type is not None:
        before = len(df)
        df = df[
            (df["flanking_bp_type_5p"] == bp_type)
            | (df["flanking_bp_type_3p"] == bp_type)
        ]
        filters_applied.append(f"bp_type={bp_type}: {before} -> {len(df)}")

    # Sequence filters
    if exclude_modified:
        before = len(df)
        df = df[~df["sequence"].str.contains("X")]
        filters_applied.append(f"exclude_modified: {before} -> {len(df)}")

    # Unpaired A+C filters
    if min_unpaired_ac is not None:
        before = len(df)
        df = df[df["unpaired_AC"] >= min_unpaired_ac]
        filters_applied.append(f"min_unpaired_ac={min_unpaired_ac}: {before} -> {len(df)}")

    if max_unpaired_ac is not None:
        before = len(df)
        df = df[df["unpaired_AC"] <= max_unpaired_ac]
        filters_applied.append(f"max_unpaired_ac={max_unpaired_ac}: {before} -> {len(df)}")

    # Add counts before deduplication
    if add_counts:
        counts = df[dedupe_by].value_counts()
        df["count"] = df[dedupe_by].map(counts)

    # Deduplication
    if not keep_duplicates:
        before = len(df)
        # Keep the first occurrence (could also keep by best resolution, etc.)
        df = df.drop_duplicates(subset=[dedupe_by], keep="first")
        filters_applied.append(f"deduplicate by {dedupe_by}: {before} -> {len(df)}")

    # Sorting
    if sort_by == "sequence":
        df = df.sort_values("sequence")
    elif sort_by == "size":
        df = df.sort_values(["total_size", "strand1_size", "sequence"])
    elif sort_by == "pdb_id":
        df = df.sort_values(["pdb_id", "motif_name"])
    elif sort_by == "count" and add_counts:
        df = df.sort_values("count", ascending=False)

    # Print filter summary
    if verbose:
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        for f in filters_applied:
            print(f"  {f}")
        print("=" * 60)

    # Prepare output columns
    output_cols = [
        "motif_name",
        "pdb_id",
        "sequence",
        "structure",
        "extended_sequence",
        "extended_structure",
        "flanking_bp_type_5p",
        "flanking_bp_type_3p",
        "size_str",
        "unpaired_A",
        "unpaired_C",
        "unpaired_AC",
    ]
    if add_counts:
        output_cols.append("count")

    df_out = df[output_cols]

    # Save output
    df_out.to_csv(output_path, index=False)
    log.info(f"Saved {len(df_out)} motifs to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input:  {initial_count} motifs")
    print(f"  Output: {len(df_out)} motifs")
    print(f"  Reduction: {initial_count - len(df_out)} ({(initial_count - len(df_out))/initial_count*100:.1f}%)")

    # Size distribution
    print("\nSize distribution:")
    size_counts = df_out["size_str"].value_counts().head(10)
    for size, count in size_counts.items():
        print(f"  {size}: {count}")
    if len(df_out["size_str"].unique()) > 10:
        print(f"  ... and {len(df_out['size_str'].unique()) - 10} more sizes")

    # Flanking BP distribution
    print("\n5' flanking BP types:")
    bp5_counts = df_out["flanking_bp_type_5p"].value_counts()
    for bp, count in bp5_counts.items():
        label = bp if bp else "(none)"
        print(f"  {label}: {count}")

    print("\n3' flanking BP types:")
    bp3_counts = df_out["flanking_bp_type_3p"].value_counts()
    for bp, count in bp3_counts.items():
        label = bp if bp else "(none)"
        print(f"  {label}: {count}")

    # Unpaired A+C distribution
    print("\nUnpaired A+C counts:")
    ac_counts = df_out["unpaired_AC"].value_counts().sort_index()
    for ac, count in ac_counts.items():
        print(f"  {ac}: {count}")
    print(f"\n  Mean unpaired A+C: {df_out['unpaired_AC'].mean():.1f}")
    print(f"  Total unpaired A: {df_out['unpaired_A'].sum()}")
    print(f"  Total unpaired C: {df_out['unpaired_C'].sum()}")

    print("=" * 60)


if __name__ == "__main__":
    filter_motifs()
