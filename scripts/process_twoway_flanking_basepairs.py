"""
Process TWOWAY motifs to find flanking basepairs and generate extended sequences/structures.

This script analyzes TWOWAY motifs from the non-redundant set and identifies flanking
basepairs at the 5' and 3' ends. It generates extended sequences and dot-bracket
structures that include the flanking basepairs.

Usage:
    python scripts/process_twoway_flanking_basepairs.py process -p 10 -o twoway_flanking_basepairs.csv
"""

import os
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd

from rna_motif_library.basepair import Basepair, get_cached_basepairs
from rna_motif_library.chain import Chains, get_cached_chains
from rna_motif_library.logger import get_logger
from rna_motif_library.motif import Motif, get_cached_motifs
from rna_motif_library.parallel_utils import run_w_processes_in_batches
from rna_motif_library.settings import DATA_PATH

log = get_logger("process_twoway_flanking")


def build_basepair_lookup(basepairs: List[Basepair]) -> Dict[str, Basepair]:
    """
    Build a lookup dictionary for basepairs with both residue orderings.

    Args:
        basepairs: List of Basepair objects

    Returns:
        Dict mapping "res1_str-res2_str" to Basepair (both directions)
    """
    bp_lookup = {}
    for bp in basepairs:
        key1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
        key2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
        bp_lookup[key1] = bp
        bp_lookup[key2] = bp
    return bp_lookup


def find_motif_by_name(motifs: List[Motif], name: str) -> Optional[Motif]:
    """
    Find a motif by its name.

    Args:
        motifs: List of Motif objects
        name: Motif name to search for

    Returns:
        Matching Motif or None if not found
    """
    for motif in motifs:
        if motif.name == name:
            return motif
    return None


def get_flanking_residues(motif: Motif, chains: Chains) -> dict:
    """
    Get the flanking residues for a TWOWAY motif.

    For TWOWAY motifs with 2 strands in antiparallel arrangement:
    - Strand 0: 5' to 3' direction
    - Strand 1: 3' to 5' direction (antiparallel)

    Returns dict with:
        prev_0: residue before first residue of strand 0
        next_0: residue after last residue of strand 0
        prev_1: residue before first residue of strand 1
        next_1: residue after last residue of strand 1
    """
    strand_0 = motif.strands[0]
    strand_1 = motif.strands[1]

    # Get chain residues for each motif residue
    first_0 = chains.get_residue_by_str(strand_0[0].get_str())
    last_0 = chains.get_residue_by_str(strand_0[-1].get_str())
    first_1 = chains.get_residue_by_str(strand_1[0].get_str())
    last_1 = chains.get_residue_by_str(strand_1[-1].get_str())

    return {
        "prev_0": chains.get_previous_residue_in_chain(first_0) if first_0 else None,
        "next_0": chains.get_next_residue_in_chain(last_0) if last_0 else None,
        "prev_1": chains.get_previous_residue_in_chain(first_1) if first_1 else None,
        "next_1": chains.get_next_residue_in_chain(last_1) if last_1 else None,
    }


def find_flanking_basepair(
    res1, res2, bp_lookup: Dict[str, Basepair]
) -> Optional[Basepair]:
    """
    Check if two residues form a basepair.

    Args:
        res1: First residue
        res2: Second residue
        bp_lookup: Basepair lookup dictionary

    Returns:
        Basepair if found, None otherwise
    """
    if res1 is None or res2 is None:
        return None
    key = f"{res1.get_str()}-{res2.get_str()}"
    return bp_lookup.get(key)


def generate_dot_bracket(motif: Motif) -> str:
    """
    Generate dot-bracket notation for a TWOWAY motif.

    For TWOWAY motifs, the structure is:
    - '(' for residues in strand 0 that have a basepair to strand 1
    - ')' for residues in strand 1 that have a basepair to strand 0
    - '.' for unpaired residues

    Returns:
        Dot-bracket string with '&' separating strands
    """
    strand_0 = motif.strands[0]
    strand_1 = motif.strands[1]

    # Build lookup for basepaired residues
    bp_residues = {}
    for bp in motif.basepairs:
        bp_residues[bp.res_1.get_str()] = bp.res_2.get_str()
        bp_residues[bp.res_2.get_str()] = bp.res_1.get_str()

    # Also include basepair_ends
    for bp in motif.basepair_ends:
        bp_residues[bp.res_1.get_str()] = bp.res_2.get_str()
        bp_residues[bp.res_2.get_str()] = bp.res_1.get_str()

    # Build structure for strand 0
    struct_0 = ""
    strand_1_strs = {res.get_str() for res in strand_1}
    for res in strand_0:
        res_str = res.get_str()
        if res_str in bp_residues and bp_residues[res_str] in strand_1_strs:
            struct_0 += "("
        else:
            struct_0 += "."

    # Build structure for strand 1
    struct_1 = ""
    strand_0_strs = {res.get_str() for res in strand_0}
    for res in strand_1:
        res_str = res.get_str()
        if res_str in bp_residues and bp_residues[res_str] in strand_0_strs:
            struct_1 += ")"
        else:
            struct_1 += "."

    return f"{struct_0}&{struct_1}"


def get_sequence(motif: Motif) -> str:
    """Get sequence string for motif with strands separated by '-'."""
    seqs = []
    for strand in motif.strands:
        seq = ""
        for res in strand:
            if res.res_id in ["A", "G", "C", "U"]:
                seq += res.res_id
            else:
                seq += "X"
        seqs.append(seq)
    return "-".join(seqs)


def process_twoway_motif(motif_row: dict) -> Optional[dict]:
    """
    Process a single TWOWAY motif to find flanking basepairs.

    Args:
        motif_row: Dictionary with motif_name and pdb_id

    Returns:
        Dictionary with motif analysis results or None on error
    """
    motif_name = motif_row["motif_name"]
    pdb_id = motif_row["pdb_id"]

    try:
        # Load cached data
        motifs = get_cached_motifs(pdb_id)
        chains_list = get_cached_chains(pdb_id)
        chains = Chains(chains_list)
        basepairs = get_cached_basepairs(pdb_id)

        # Build basepair lookup
        bp_lookup = build_basepair_lookup(basepairs)

        # Find the specific motif
        motif = find_motif_by_name(motifs, motif_name)
        if motif is None:
            log.warning(f"Motif {motif_name} not found in {pdb_id}")
            return None

        if len(motif.strands) != 2:
            log.warning(f"Motif {motif_name} does not have 2 strands")
            return None

        # Get original sequence and structure
        sequence = get_sequence(motif)
        structure = generate_dot_bracket(motif)

        # Get flanking residues
        flanking = get_flanking_residues(motif, chains)

        # Check for flanking basepairs (antiparallel: prev_0 pairs with next_1, next_0 pairs with prev_1)
        bp_5p = find_flanking_basepair(flanking["prev_0"], flanking["next_1"], bp_lookup)
        bp_3p = find_flanking_basepair(flanking["next_0"], flanking["prev_1"], bp_lookup)

        # Build extended sequence and structure
        strand_0_seq = sequence.split("-")[0]
        strand_1_seq = sequence.split("-")[1]
        struct_0 = structure.split("&")[0]
        struct_1 = structure.split("&")[1]

        ext_strand_0_seq = strand_0_seq
        ext_strand_1_seq = strand_1_seq
        ext_struct_0 = struct_0
        ext_struct_1 = struct_1

        flanking_bp_type_5p = ""
        flanking_bp_type_3p = ""

        # Add 5' flanking basepair (prepend to strand 0, append to strand 1)
        if bp_5p is not None:
            prev_0_id = flanking["prev_0"].res_id if flanking["prev_0"].res_id in "AGCU" else "X"
            next_1_id = flanking["next_1"].res_id if flanking["next_1"].res_id in "AGCU" else "X"
            ext_strand_0_seq = prev_0_id + ext_strand_0_seq
            ext_strand_1_seq = ext_strand_1_seq + next_1_id
            ext_struct_0 = "(" + ext_struct_0
            ext_struct_1 = ext_struct_1 + ")"
            flanking_bp_type_5p = bp_5p.lw

        # Add 3' flanking basepair (append to strand 0, prepend to strand 1)
        if bp_3p is not None:
            next_0_id = flanking["next_0"].res_id if flanking["next_0"].res_id in "AGCU" else "X"
            prev_1_id = flanking["prev_1"].res_id if flanking["prev_1"].res_id in "AGCU" else "X"
            ext_strand_0_seq = ext_strand_0_seq + next_0_id
            ext_strand_1_seq = prev_1_id + ext_strand_1_seq
            ext_struct_0 = ext_struct_0 + "("
            ext_struct_1 = ")" + ext_struct_1

            flanking_bp_type_3p = bp_3p.lw

        extended_sequence = f"{ext_strand_0_seq}-{ext_strand_1_seq}"
        extended_structure = f"{ext_struct_0}&{ext_struct_1}"

        return {
            "motif_name": motif_name,
            "pdb_id": pdb_id,
            "sequence": sequence,
            "structure": structure,
            "extended_sequence": extended_sequence,
            "extended_structure": extended_structure,
            "flanking_bp_type_5p": flanking_bp_type_5p,
            "flanking_bp_type_3p": flanking_bp_type_3p,
        }

    except FileNotFoundError as e:
        log.warning(f"Cache files not found for {pdb_id}: {e}")
        return None
    except Exception as e:
        log.error(f"Error processing {motif_name}: {e}")
        return None


@click.group()
def cli():
    """Process TWOWAY motifs to find flanking basepairs."""
    pass


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    default=None,
    help="Input CSV path (default: data/summaries/non_redundant_motifs_no_issues.csv)",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="twoway_flanking_basepairs.csv",
    help="Output CSV path",
)
@click.option(
    "-p",
    "--processes",
    default=10,
    type=int,
    help="Number of parallel processes",
)
@click.option(
    "-c",
    "--chunk-size",
    default=1000,
    type=int,
    help="Number of motifs per chunk (saves after each chunk)",
)
def process(input_path, output_path, processes, chunk_size):
    """Process all TWOWAY motifs and find flanking basepairs."""
    # Set default input path
    if input_path is None:
        input_path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")

    # Read input CSV
    log.info(f"Reading input from {input_path}")
    df = pd.read_csv(input_path)

    # Filter to only TWOWAY motifs
    df_twoway = df[df["motif_type"] == "TWOWAY"].copy()
    total_motifs = len(df_twoway)
    log.info(f"Found {total_motifs} TWOWAY motifs")

    # Prepare rows for processing
    motif_rows = df_twoway[["motif_name", "pdb_id"]].to_dict("records")

    # Process in chunks with progress updates
    all_results = []
    num_chunks = (total_motifs + chunk_size - 1) // chunk_size

    log.info(f"Processing in {num_chunks} chunks of {chunk_size} motifs each")
    log.info(f"Using {processes} parallel processes")
    print("-" * 60)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_motifs)
        chunk = motif_rows[start_idx:end_idx]

        print(f"\n[Chunk {chunk_idx + 1}/{num_chunks}] Processing motifs {start_idx + 1}-{end_idx} of {total_motifs}")

        # Process chunk in parallel
        chunk_results = run_w_processes_in_batches(
            items=chunk,
            func=process_twoway_motif,
            processes=processes,
            batch_size=100,
            desc=f"Chunk {chunk_idx + 1}",
        )

        # Filter None results
        chunk_results = [r for r in chunk_results if r is not None]
        all_results.extend(chunk_results)

        # Progress update
        success_rate = len(chunk_results) / len(chunk) * 100
        print(f"[Chunk {chunk_idx + 1}/{num_chunks}] Completed: {len(chunk_results)}/{len(chunk)} ({success_rate:.1f}% success)")
        print(f"[Progress] Total processed so far: {len(all_results)} motifs ({len(all_results)/total_motifs*100:.1f}%)")

        # Save intermediate results
        if all_results:
            df_intermediate = pd.DataFrame(all_results)
            intermediate_path = output_path.replace(".csv", "_intermediate.csv")
            df_intermediate.to_csv(intermediate_path, index=False)
            print(f"[Saved] Intermediate results to {intermediate_path}")

        print("-" * 60)

    log.info(f"Successfully processed {len(all_results)} motifs total")

    # Create final DataFrame
    df_results = pd.DataFrame(all_results)

    # Reorder columns
    column_order = [
        "motif_name",
        "pdb_id",
        "sequence",
        "structure",
        "extended_sequence",
        "extended_structure",
        "flanking_bp_type_5p",
        "flanking_bp_type_3p",
    ]
    df_results = df_results[column_order]

    # Save final results
    df_results.to_csv(output_path, index=False)
    log.info(f"Final results saved to {output_path}")

    # Remove intermediate file
    intermediate_path = output_path.replace(".csv", "_intermediate.csv")
    if os.path.exists(intermediate_path):
        os.remove(intermediate_path)
        log.info(f"Removed intermediate file {intermediate_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    has_5p = (df_results["flanking_bp_type_5p"] != "").sum()
    has_3p = (df_results["flanking_bp_type_3p"] != "").sum()
    has_both = ((df_results["flanking_bp_type_5p"] != "") & (df_results["flanking_bp_type_3p"] != "")).sum()

    print(f"  Total processed: {len(df_results)}")
    print(f"  With 5' flanking BP: {has_5p} ({has_5p/len(df_results)*100:.1f}%)")
    print(f"  With 3' flanking BP: {has_3p} ({has_3p/len(df_results)*100:.1f}%)")
    print(f"  With both flanking BPs: {has_both} ({has_both/len(df_results)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    cli()
