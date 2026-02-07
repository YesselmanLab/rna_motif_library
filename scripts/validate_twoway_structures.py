"""
Validate TWOWAY motif structures by embedding in a hairpin and folding with ViennaRNA.

This script takes filtered TWOWAY motifs and validates that their secondary structure
is correctly predicted when embedded in a proper hairpin context with random helices.

Based on the approach from twoway-lib-generation.

Usage:
    python scripts/validate_twoway_structures.py -i filtered.csv -o validated.csv

    # With custom parameters
    python scripts/validate_twoway_structures.py -i filtered.csv -o validated.csv --helix-len 3 --repeats 5
"""

from dataclasses import dataclass
from random import Random

import click
import pandas as pd
import RNA

from rna_motif_library.logger import get_logger

log = get_logger("validate_structures")

# Watson-Crick base pairs
WC_PAIRS = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]


@dataclass
class FoldResult:
    """Result of motif fold validation."""
    motif_name: str
    pdb_id: str
    sequence: str
    extended_sequence: str
    expected_structure: str
    predicted_structure: str
    full_sequence: str
    full_structure: str
    mfe: float
    match: bool
    accuracy: float
    instances_matched: int
    total_instances: int


def random_helix(length: int, rng: Random) -> tuple:
    """
    Generate a random Watson-Crick helix.

    Returns:
        Tuple of (strand1, strand2, structure1, structure2)
    """
    bp_combo = [rng.choice(WC_PAIRS) for _ in range(length)]
    strand1 = "".join(bp[0] for bp in bp_combo)
    strand2 = "".join(bp[1] for bp in reversed(bp_combo))
    structure1 = "(" * length
    structure2 = ")" * length
    return strand1, strand2, structure1, structure2


def build_test_construct(
    strand1_seq: str,
    strand1_ss: str,
    strand2_seq: str,
    strand2_ss: str,
    helix_len: int,
    repeats: int,
    rng: Random,
) -> tuple:
    """
    Build a test construct with the motif embedded multiple times.

    Layout (5' to 3'):
    Helix - Motif_s1 - Helix - Motif_s1 - ... - Helix - GAAA - Helix - ... - Motif_s2 - Helix - Motif_s2 - Helix

    Returns:
        Tuple of (sequence, structure, motif_positions)
        motif_positions is list of (strand1_start, strand1_end, strand2_start, strand2_end)
    """
    # Generate random helices
    helices = [random_helix(helix_len, rng) for _ in range(repeats + 1)]

    seq_parts = []
    ss_parts = []
    motif_positions = []

    current_pos = 0

    # Build 5' arm: Helix - Motif_s1 - Helix - Motif_s1 - ... - Helix
    for i in range(repeats):
        # Add helix strand1
        h_s1, _, h_ss1, _ = helices[i]
        seq_parts.append(h_s1)
        ss_parts.append(h_ss1)
        current_pos += helix_len

        # Track motif strand1 position
        s1_start = current_pos
        seq_parts.append(strand1_seq)
        ss_parts.append(strand1_ss)
        current_pos += len(strand1_seq)
        s1_end = current_pos

        motif_positions.append([s1_start, s1_end, None, None])

    # Add final helix strand1
    h_s1, _, h_ss1, _ = helices[repeats]
    seq_parts.append(h_s1)
    ss_parts.append(h_ss1)
    current_pos += helix_len

    # Add hairpin loop
    seq_parts.append("GAAA")
    ss_parts.append("....")
    current_pos += 4

    # Build 3' arm: Helix - Motif_s2 - Helix - ... (in reverse order)
    # Add final helix strand2
    _, h_s2, _, h_ss2 = helices[repeats]
    seq_parts.append(h_s2)
    ss_parts.append(h_ss2)
    current_pos += helix_len

    # Add motifs in reverse order
    for i in range(repeats - 1, -1, -1):
        # Track motif strand2 position
        s2_start = current_pos
        seq_parts.append(strand2_seq)
        ss_parts.append(strand2_ss)
        current_pos += len(strand2_seq)
        s2_end = current_pos

        motif_positions[i][2] = s2_start
        motif_positions[i][3] = s2_end

        # Add helix strand2
        _, h_s2, _, h_ss2 = helices[i]
        seq_parts.append(h_s2)
        ss_parts.append(h_ss2)
        current_pos += helix_len

    sequence = "".join(seq_parts)
    structure = "".join(ss_parts)

    return sequence, structure, motif_positions


def fold_sequence(sequence: str) -> tuple:
    """Fold an RNA sequence using ViennaRNA."""
    fc = RNA.fold_compound(sequence)
    structure, mfe = fc.mfe()
    return structure, mfe


def validate_motif(row: dict, helix_len: int, repeats: int, seed: int) -> FoldResult:
    """
    Validate a single motif by embedding in a hairpin construct.

    Args:
        row: DataFrame row with motif data
        helix_len: Length of flanking helices
        repeats: Number of times to repeat the motif
        seed: Random seed for helix generation

    Returns:
        FoldResult with validation results
    """
    rng = Random(seed)

    extended_seq = row["extended_sequence"]
    expected_struct = row["extended_structure"]

    # Parse motif strands
    parts = extended_seq.split("-")
    if len(parts) != 2:
        return FoldResult(
            motif_name=row["motif_name"],
            pdb_id=row["pdb_id"],
            sequence=row.get("sequence", ""),
            extended_sequence=extended_seq,
            expected_structure=expected_struct,
            predicted_structure="",
            full_sequence="",
            full_structure="",
            mfe=0.0,
            match=False,
            accuracy=0.0,
            instances_matched=0,
            total_instances=0,
        )

    strand1_seq, strand2_seq = parts

    # Parse structure strands
    struct_parts = expected_struct.split("&")
    if len(struct_parts) != 2:
        struct_parts = [expected_struct[:len(strand1_seq)], expected_struct[len(strand1_seq):]]

    strand1_ss, strand2_ss = struct_parts

    # Build test construct
    full_seq, designed_ss, motif_positions = build_test_construct(
        strand1_seq, strand1_ss,
        strand2_seq, strand2_ss,
        helix_len, repeats, rng
    )

    # Fold with ViennaRNA
    predicted_ss, mfe = fold_sequence(full_seq)

    # Check each motif instance
    instances_matched = 0
    total_mismatches = 0
    total_positions = 0

    # Get predicted structure for first instance (for reporting)
    first_s1_start, first_s1_end, first_s2_start, first_s2_end = motif_positions[0]
    pred_s1 = predicted_ss[first_s1_start:first_s1_end]
    pred_s2 = predicted_ss[first_s2_start:first_s2_end]
    predicted_motif_struct = f"{pred_s1}&{pred_s2}"

    for s1_start, s1_end, s2_start, s2_end in motif_positions:
        instance_match = True

        # Check strand1
        for j in range(s1_end - s1_start):
            pos = s1_start + j
            total_positions += 1
            if designed_ss[pos] != predicted_ss[pos]:
                total_mismatches += 1
                instance_match = False

        # Check strand2
        for j in range(s2_end - s2_start):
            pos = s2_start + j
            total_positions += 1
            if designed_ss[pos] != predicted_ss[pos]:
                total_mismatches += 1
                instance_match = False

        if instance_match:
            instances_matched += 1

    accuracy = (total_positions - total_mismatches) / total_positions if total_positions > 0 else 0.0
    match = accuracy == 1.0

    return FoldResult(
        motif_name=row["motif_name"],
        pdb_id=row["pdb_id"],
        sequence=row.get("sequence", ""),
        extended_sequence=extended_seq,
        expected_structure=expected_struct,
        predicted_structure=predicted_motif_struct,
        full_sequence=full_seq,
        full_structure=predicted_ss,
        mfe=mfe,
        match=match,
        accuracy=accuracy,
        instances_matched=instances_matched,
        total_instances=repeats,
    )


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input CSV from filter_twoway_motifs.py",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    help="Output CSV path",
)
@click.option(
    "--helix-len",
    type=int,
    default=5,
    help="Length of flanking helices in base pairs (default: 5)",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    help="Number of times to repeat motif in test construct (default: 5)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for helix generation (default: 42)",
)
@click.option(
    "--only-matches",
    is_flag=True,
    help="Only output motifs where predicted matches expected",
)
@click.option(
    "--only-mismatches",
    is_flag=True,
    help="Only output motifs where predicted differs from expected",
)
@click.option(
    "--min-accuracy",
    type=float,
    default=None,
    help="Minimum accuracy threshold (0.0-1.0)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed progress",
)
def validate(
    input_path,
    output_path,
    helix_len,
    repeats,
    seed,
    only_matches,
    only_mismatches,
    min_accuracy,
    verbose,
):
    """Validate TWOWAY structures by embedding in hairpin and folding."""
    # Load input
    log.info(f"Reading {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df)} motifs")
    log.info(f"Helix length: {helix_len}, Repeats: {repeats}, Seed: {seed}")

    # Validate each motif
    results = []
    for idx, row in df.iterrows():
        if verbose and idx % 100 == 0:
            print(f"Processing {idx + 1}/{len(df)}...")

        result = validate_motif(row.to_dict(), helix_len, repeats, seed)
        results.append({
            "motif_name": result.motif_name,
            "pdb_id": result.pdb_id,
            "sequence": result.sequence,
            "extended_sequence": result.extended_sequence,
            "expected_structure": result.expected_structure,
            "predicted_structure": result.predicted_structure,
            "full_sequence": result.full_sequence,
            "full_structure": result.full_structure,
            "mfe": result.mfe,
            "match": result.match,
            "accuracy": result.accuracy,
            "instances_matched": result.instances_matched,
            "total_instances": result.total_instances,
        })

    df_results = pd.DataFrame(results)

    # Filter results
    if only_matches:
        df_results = df_results[df_results["match"] == True]
        log.info(f"Filtered to {len(df_results)} matching motifs")

    if only_mismatches:
        df_results = df_results[df_results["match"] == False]
        log.info(f"Filtered to {len(df_results)} mismatching motifs")

    if min_accuracy is not None:
        df_results = df_results[df_results["accuracy"] >= min_accuracy]
        log.info(f"Filtered to {len(df_results)} motifs with accuracy >= {min_accuracy}")

    # Save results
    df_results.to_csv(output_path, index=False)
    log.info(f"Saved {len(df_results)} results to {output_path}")

    # Print summary
    total = len(df_results)
    if total > 0:
        matches = df_results["match"].sum()
        mean_accuracy = df_results["accuracy"].mean()

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  Total motifs: {len(df)}")
        print(f"  Exact matches: {matches} ({matches/total*100:.1f}%)")
        print(f"  Mismatches: {total - matches} ({(total-matches)/total*100:.1f}%)")
        print(f"  Mean accuracy: {mean_accuracy:.3f}")

        # Accuracy distribution
        print("\nAccuracy distribution:")
        bins = [0, 0.5, 0.8, 0.9, 0.95, 1.0]
        for i in range(len(bins) - 1):
            count = ((df_results["accuracy"] >= bins[i]) &
                     (df_results["accuracy"] < bins[i+1])).sum()
            print(f"  {bins[i]:.0%}-{bins[i+1]:.0%}: {count}")
        perfect = (df_results["accuracy"] == 1.0).sum()
        print(f"  100%: {perfect}")

        # Show some examples of mismatches
        mismatches = df_results[df_results["match"] == False]
        if len(mismatches) > 0:
            print("\nExample mismatches (first 5):")
            for _, row in mismatches.head(5).iterrows():
                print(f"\n  {row['motif_name']}")
                print(f"    Sequence: {row['extended_sequence']}")
                print(f"    Expected:  {row['expected_structure']}")
                print(f"    Predicted: {row['predicted_structure']}")
                print(f"    Accuracy: {row['accuracy']:.1%}")
                print(f"    Instances: {row['instances_matched']}/{row['total_instances']}")

        print("=" * 60)


if __name__ == "__main__":
    validate()
