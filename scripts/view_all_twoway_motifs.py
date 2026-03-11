"""
View ALL curated TWOWAY motifs organized by topology in PyMOL.

Loads curated_twoway.json, groups by topology, aligns motifs within
each topology group by C1' superposition, and generates a PyMOL session.

Usage:
    python scripts/view_all_twoway_motifs.py -o all_twoway_view

    # Then open in PyMOL:
    pymol all_twoway_view/view.pml
"""

import copy
import os

import click
import numpy as np
import pandas as pd

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.tranforms import kabsch_algorithm

log = get_logger("view_all_twoway")

COLORS = [
    "green", "cyan", "magenta", "yellow", "salmon", "lightorange",
    "palegreen", "lightblue", "lightpink", "palecyan", "lightteal",
    "wheat", "violet", "slate", "yelloworange", "tv_yellow",
    "chartreuse", "aquamarine", "tv_blue", "pink", "tv_orange",
    "lime", "limon", "gray60", "deepblue", "deeppurple",
    "ruby", "chocolate", "teal", "forest", "firebrick",
    "smudge", "dirtyviolet", "deepsalmon", "lightmagenta",
    "brightorange", "splitpea", "raspberry", "sand", "warmpink",
    "darksalmon",
]


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


def safe_name(motif_id):
    return motif_id.replace("/", "_")


def _parse_topo_sort_key(topo):
    """Sort topologies numerically: '1-0' < '1-1' < '2-0' etc."""
    parts = topo.split("-")
    return (int(parts[0]), int(parts[1]))


@click.command()
@click.option(
    "-o", "--output", "output_dir", required=True,
    help="Output directory for CIF files and PyMOL script",
)
@click.option(
    "--curated-path", default=None,
    help="Path to curated JSON (default: data/summaries/motifs/curated_twoway.json)",
)
def main(output_dir, curated_path):
    """View all curated TWOWAY motifs organized by topology."""
    if curated_path is None:
        curated_path = os.path.join(
            DATA_PATH, "summaries", "motifs", "curated_twoway.json"
        )

    print(f"Loading {curated_path}")
    df = pd.read_json(curated_path)
    print(f"Total curated motifs: {len(df)}")

    os.makedirs(output_dir, exist_ok=True)
    motif_cache = {}
    abs_output_dir = os.path.abspath(output_dir)

    # Sort topologies numerically
    topos = sorted(df["motif_topology"].unique(), key=_parse_topo_sort_key)

    pml_lines = []
    pml_lines.append("# Auto-generated PyMOL script: ALL curated TWOWAY motifs")
    pml_lines.append(f"# Total motifs: {len(df)}")
    pml_lines.append(f"# Topologies: {len(topos)}")
    pml_lines.append("")
    pml_lines.append(f"cd {abs_output_dir}")
    pml_lines.append("")

    color_idx = 0
    total_exported = 0
    failed = 0

    for topo in topos:
        group = df[df["motif_topology"] == topo].sort_values(
            ["resolution"] if "resolution" in df.columns else ["motif_id"]
        )
        n_motifs = len(group)
        has_bp = (group["bp_sig"] != "").sum() if "bp_sig" in group.columns else "?"

        print(f"\nTopology {topo}: {n_motifs} motif(s)")
        pml_lines.append(f"# === Topology {topo} ({n_motifs} motifs) ===")

        # Load first motif as alignment reference
        ref_row = group.iloc[0]
        ref_motif = load_motif(ref_row["motif_id"], ref_row["pdb_id"], motif_cache)
        if ref_motif is None:
            print(f"  Could not load reference {ref_row['motif_id']}, skipping topology")
            failed += n_motifs
            continue

        topo_member_names = []

        for i, (_, row) in enumerate(group.iterrows()):
            motif_id = row["motif_id"]
            pdb_id = row["pdb_id"]
            seq = row.get("motif_sequence", "?")
            bp_sig = row.get("bp_sig", "")
            if not bp_sig:
                bp_sig = "(none)"
            res = row.get("resolution", 99.0)

            motif = load_motif(motif_id, pdb_id, motif_cache)
            if motif is None:
                print(f"  Could not load {motif_id}")
                failed += 1
                continue

            if i == 0:
                aligned = motif
            else:
                aligned = align_motif_to_target(motif, ref_motif)
                if aligned is None:
                    aligned = motif

            name = safe_name(motif_id)
            cif_path = os.path.join(output_dir, f"{name}.cif")
            aligned.to_cif(cif_path)

            color = COLORS[color_idx % len(COLORS)]
            label = f"seq={seq}  bp_sig={bp_sig}  res={res:.2f}"

            pml_lines.append(f"# {label}")
            pml_lines.append(f"load {name}.cif, {name}")
            pml_lines.append(f"color {color}, {name}")

            topo_member_names.append(name)
            color_idx += 1
            total_exported += 1

        if topo_member_names:
            group_name = f"topo_{topo.replace('-', 'x')}"
            pml_lines.append(
                f"group {group_name}, {' '.join(topo_member_names)}"
            )
        pml_lines.append("")

    # Final PyMOL setup
    pml_lines.append("# Display settings")
    pml_lines.append("hide everything")
    pml_lines.append("show sticks")
    pml_lines.append("set stick_radius, 0.15")
    pml_lines.append("set ray_opaque_background, 0")
    pml_lines.append("zoom all")
    pml_lines.append("")
    pml_lines.append("# Toggle topology groups on/off:")
    for topo in topos:
        group_name = f"topo_{topo.replace('-', 'x')}"
        pml_lines.append(f"#   disable {group_name}")

    pml_path = os.path.join(output_dir, "view.pml")
    with open(pml_path, "w") as f:
        f.write("\n".join(pml_lines) + "\n")

    print(f"\nExported {total_exported} CIF files ({failed} failed) to {output_dir}/")
    print(f"PyMOL script: {pml_path}")
    print(f"\nOpen in PyMOL: pymol {pml_path}")


if __name__ == "__main__":
    main()
