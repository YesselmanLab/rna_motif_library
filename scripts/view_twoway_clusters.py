"""
View TWOWAY motif clusters in PyMOL.

Loads a representative motif and all rejected motifs that map to it,
aligns them by C1' superposition, exports CIF files, and generates
a PyMOL script for visualization.

Usage:
    # View all motifs mapping to a specific representative
    python scripts/view_twoway_clusters.py -t tracking.json -r TWOWAY-1-1-GAC-GCC-7ECN-1 -o cluster_view

    # View all kept motifs for a topology
    python scripts/view_twoway_clusters.py -t tracking.json --topology 3-3 -o topo_view

    # View with specific rejection reason filter
    python scripts/view_twoway_clusters.py -t tracking.json -r TWOWAY-1-1-GAC-GCC-7ECN-1 -o cluster_view --reason sequence_dedup

    # Then open in PyMOL:
    pymol cluster_view/view.pml
"""

import copy
import json
import os

import click
import numpy as np
import pandas as pd

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import Motif, get_cached_motifs
from rna_motif_library.tranforms import kabsch_algorithm

log = get_logger("view_clusters")

# Color palette for PyMOL (representative gets green, rest get these)
COLORS = [
    "gray60", "salmon", "lightorange", "palegreen", "lightblue",
    "lightpink", "palecyan", "lightteal", "wheat", "violet",
    "slate", "yelloworange", "tv_yellow", "chartreuse", "aquamarine",
    "tv_blue", "pink", "tv_orange", "lime", "limon",
]


def _safe(val):
    """Return empty string for NaN/None values."""
    if pd.isna(val):
        return ""
    return val


def _parse_json_col(val):
    """Parse a JSON column, returning [] for empty/NaN."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def format_basepairs(bp_json):
    """Format basepairs JSON like [['AG','cWW']] to 'AG:cWW'."""
    bps = _parse_json_col(bp_json)
    if not bps:
        return "none"
    return " ".join(f"{bp[0]}:{bp[1]}" for bp in bps)


def motif_summary_str(row):
    """Build a one-line summary of a motif from its tracking row."""
    parts = [
        f"seq={row['motif_sequence']}",
        f"topo={row['motif_topology']}",
    ]
    bp_str = format_basepairs(row.get("basepairs", "[]"))
    if bp_str != "none":
        parts.append(f"bps={bp_str}")
    parts.append(f"res={row.get('resolution', '?')}")
    n_prot = row.get("num_protein_hbonds", 0)
    n_lig = row.get("num_ligand_hbonds", 0)
    if n_prot > 0:
        parts.append(f"prot_hb={n_prot}")
    if n_lig > 0:
        parts.append(f"lig_hb={n_lig}")
    if row.get("in_tertiary_contact", 0):
        parts.append(f"tc={row.get('num_tertiary_contacts', 0)}")
    if row.get("has_non_canonical_residue", 0):
        parts.append("mod_nuc")
    if row.get("is_bulge", False):
        parts.append("bulge")
    return "  ".join(parts)


def print_motif_details(row, indent="  "):
    """Print detailed motif info to terminal."""
    print(f"{indent}sequence:   {row['motif_sequence']}  "
          f"(s1={_safe(row.get('strand1_sequence', ''))} "
          f"s2={_safe(row.get('strand2_sequence', ''))})")
    print(f"{indent}topology:   {row['motif_topology']}  "
          f"bulge={row.get('is_bulge', '?')}")
    print(f"{indent}basepairs:  {format_basepairs(row.get('basepairs', '[]'))}")
    print(f"{indent}resolution: {row.get('resolution', '?')}  "
          f"residues={row.get('num_residues', '?')}")
    hb = row.get("num_hbonds", "?")
    phb = row.get("num_protein_hbonds", 0)
    lhb = row.get("num_ligand_hbonds", 0)
    print(f"{indent}hbonds:     total={hb}  protein={phb}  ligand={lhb}")
    tc = row.get("in_tertiary_contact", 0)
    ntc = row.get("num_tertiary_contacts", 0)
    nc = row.get("has_non_canonical_residue", 0)
    iso = row.get("is_isolatable", "?")
    print(f"{indent}flags:      in_tc={tc}  num_tc={ntc}  "
          f"mod_nuc={nc}  isolatable={iso}")


def load_motif(motif_id, pdb_id, motif_cache):
    """Load a Motif object from cached data."""
    if pdb_id not in motif_cache:
        try:
            motif_list = get_cached_motifs(pdb_id)
            motif_cache[pdb_id] = {m.name: m for m in motif_list}
        except Exception:
            motif_cache[pdb_id] = {}
    return motif_cache[pdb_id].get(motif_id)


def align_motif_to_target(mobile_motif, target_motif):
    """Align mobile motif to target using C1' atoms."""
    coords_m = mobile_motif.get_c1prime_coords()
    coords_t = target_motif.get_c1prime_coords()
    if len(coords_m) != len(coords_t) or len(coords_m) < 3:
        return None

    rotation_matrix = kabsch_algorithm(coords_m, coords_t)
    mobile_center = np.mean(coords_m, axis=0)
    target_center = np.mean(coords_t, axis=0)

    new_m = copy.deepcopy(mobile_motif)
    for strand in new_m.strands:
        for res in strand:
            res.coords = (
                np.dot(res.coords - mobile_center, rotation_matrix) + target_center
            )
    return new_m


def write_pymol_script(output_dir, rep_name, rep_info, member_data):
    """
    Generate a PyMOL .pml script for representative + cluster view.

    Args:
        rep_info: dict with summary string for representative
        member_data: list of (name, reason, summary_str) tuples
    """
    pml_path = os.path.join(output_dir, "view.pml")
    lines = []
    lines.append("# Auto-generated PyMOL script for TWOWAY cluster viewing")
    lines.append(f"# Representative: {rep_name}")
    lines.append(f"# {rep_info}")
    lines.append(f"# Total members: {len(member_data) + 1}")
    lines.append("")

    # Load representative
    lines.append(f'load {rep_name}.cif, {rep_name}')
    lines.append(f'color green, {rep_name}')
    lines.append(f'show sticks, {rep_name}')
    lines.append("")

    # Group members by rejection reason
    reasons = {}
    for name, reason, summary in member_data:
        reasons.setdefault(reason, []).append((name, summary))

    color_idx = 0
    for reason, entries in sorted(reasons.items()):
        lines.append(f"# --- {reason} ({len(entries)} motifs) ---")
        color = COLORS[color_idx % len(COLORS)]
        group_name = f"grp_{reason}"
        group_members = []
        for name, summary in entries:
            lines.append(f'# {summary}')
            lines.append(f'load {name}.cif, {name}')
            lines.append(f'color {color}, {name}')
            lines.append(f'show sticks, {name}')
            group_members.append(name)
        lines.append(f'group {group_name}, {" ".join(group_members)}')
        lines.append("")
        color_idx += 1

    # Final setup
    lines.append("hide everything")
    lines.append("show sticks")
    lines.append("set stick_radius, 0.15")
    lines.append("set ray_opaque_background, 0")
    lines.append(f"zoom {rep_name}")
    lines.append("")
    lines.append("# Toggle groups on/off:")
    for reason in sorted(reasons.keys()):
        lines.append(f"#   disable grp_{reason}")

    with open(pml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return pml_path


def write_topology_pymol_script(output_dir, motif_entries):
    """
    Generate a PyMOL .pml script for topology view.

    Args:
        motif_entries: list of (name, summary_str) tuples
    """
    pml_path = os.path.join(output_dir, "view.pml")
    lines = []
    lines.append("# Auto-generated PyMOL script for topology view")
    lines.append(f"# {len(motif_entries)} motifs")
    lines.append("")

    for i, (name, summary) in enumerate(motif_entries):
        lines.append(f'# {summary}')
        lines.append(f'load {name}.cif, {name}')
        color = COLORS[i % len(COLORS)] if i > 0 else "green"
        lines.append(f'color {color}, {name}')
        lines.append(f'show sticks, {name}')
    lines.append("")
    lines.append("hide everything")
    lines.append("show sticks")
    lines.append("set stick_radius, 0.15")
    lines.append("set ray_opaque_background, 0")
    lines.append(f"zoom {motif_entries[0][0]}")

    with open(pml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return pml_path


def safe_name(motif_id):
    return motif_id.replace("/", "_")


@click.command()
@click.option(
    "-t", "--tracking", "tracking_path", required=True,
    type=click.Path(exists=True),
    help="Tracking JSON from filter_twoway_motifs.py",
)
@click.option(
    "-r", "--representative", "rep_id", default=None,
    help="Representative motif_id to view with all its cluster members",
)
@click.option(
    "--topology", default=None,
    help="View all kept motifs for a topology (e.g., '3-3')",
)
@click.option(
    "--reason", default=None,
    help="Filter to specific rejection reason",
)
@click.option(
    "--max-members", type=int, default=50,
    help="Maximum number of cluster members to show (default: 50)",
)
@click.option(
    "-o", "--output", "output_dir", required=True,
    help="Output directory for CIF files and PyMOL script",
)
def view_clusters(tracking_path, rep_id, topology, reason, max_members, output_dir):
    """View TWOWAY motif clusters aligned in PyMOL."""
    df = pd.read_json(tracking_path)

    if rep_id is None and topology is None:
        print("Must specify either --representative or --topology")
        return

    os.makedirs(output_dir, exist_ok=True)
    motif_cache = {}

    if topology is not None:
        # =====================================================================
        # Topology mode: show all kept motifs for a topology, aligned
        # =====================================================================
        kept = df[(df["status"] == "kept") & (df["motif_topology"] == topology)]
        if len(kept) == 0:
            print(f"No kept motifs found for topology {topology}")
            return

        kept = kept.sort_values("resolution")
        print(f"Loading {len(kept)} kept motifs for topology {topology}...")
        print()

        # Print details for each
        for i, (_, row) in enumerate(kept.iterrows()):
            tag = "(reference)" if i == 0 else ""
            print(f"  {row['motif_id']} {tag}")
            print_motif_details(row, indent="    ")
            print()

        # Load reference
        ref_row = kept.iloc[0]
        ref_motif = load_motif(ref_row["motif_id"], ref_row["pdb_id"], motif_cache)
        if ref_motif is None:
            print(f"Could not load reference motif {ref_row['motif_id']}")
            return

        ref_name = safe_name(ref_row["motif_id"])
        ref_motif.to_cif(os.path.join(output_dir, f"{ref_name}.cif"))

        motif_entries = [(ref_name, motif_summary_str(ref_row))]

        for _, row in kept.iloc[1:].iterrows():
            motif = load_motif(row["motif_id"], row["pdb_id"], motif_cache)
            if motif is None:
                continue
            aligned = align_motif_to_target(motif, ref_motif)
            if aligned is None:
                continue
            name = safe_name(row["motif_id"])
            aligned.to_cif(os.path.join(output_dir, f"{name}.cif"))
            motif_entries.append((name, motif_summary_str(row)))

        pml_path = write_topology_pymol_script(output_dir, motif_entries)
        print(f"Wrote {len(motif_entries)} CIF files and {pml_path}")
        print(f"\nOpen in PyMOL: pymol {pml_path}")
        return

    # =========================================================================
    # Representative mode: show a representative and all its cluster members
    # =========================================================================
    rep_row = df[df["motif_id"] == rep_id]
    if len(rep_row) == 0:
        print(f"Representative {rep_id} not found in tracking JSON")
        return

    rep_row = rep_row.iloc[0]
    if rep_row["status"] != "kept":
        print(f"Warning: {rep_id} is not a kept motif (status={rep_row['status']})")

    # Find all motifs that map to this representative
    members = df[
        (df["representative_motif_id"] == rep_id) & (df["motif_id"] != rep_id)
    ]
    if reason:
        members = members[members["rejection_reason"] == reason]

    print(f"Representative: {rep_id}")
    print_motif_details(rep_row)
    print(f"\nCluster members: {len(members)}")

    if len(members) > 0:
        reason_counts = members["rejection_reason"].value_counts()
        for r, c in reason_counts.items():
            print(f"  {r}: {c}")

    if len(members) > max_members:
        print(f"Showing first {max_members} members (use --max-members to change)")
        members = members.head(max_members)

    # Print member details
    if len(members) > 0:
        print()
        for _, row in members.iterrows():
            bp_str = format_basepairs(row.get("basepairs", "[]"))
            print(f"  {row['motif_id']}  [{row['rejection_reason']}]")
            print(f"    {row['motif_sequence']}  bps={bp_str}  "
                  f"res={row.get('resolution', '?')}")

    # Load representative motif
    ref_motif = load_motif(rep_id, rep_row["pdb_id"], motif_cache)
    if ref_motif is None:
        print(f"Could not load motif {rep_id}")
        return

    ref_name = safe_name(rep_id)
    ref_motif.to_cif(os.path.join(output_dir, f"{ref_name}.cif"))

    member_data = []
    print(f"\nExporting CIF files...")

    for _, row in members.iterrows():
        motif = load_motif(row["motif_id"], row["pdb_id"], motif_cache)
        if motif is None:
            continue

        aligned = align_motif_to_target(motif, ref_motif)
        if aligned is None:
            aligned = motif

        name = safe_name(row["motif_id"])
        aligned.to_cif(os.path.join(output_dir, f"{name}.cif"))
        member_data.append((name, row["rejection_reason"], motif_summary_str(row)))

    rep_info = motif_summary_str(rep_row)
    pml_path = write_pymol_script(output_dir, ref_name, rep_info, member_data)
    print(f"Wrote {len(member_data) + 1} CIF files and {pml_path}")
    print(f"\nOpen in PyMOL: pymol {pml_path}")


if __name__ == "__main__":
    view_clusters()
