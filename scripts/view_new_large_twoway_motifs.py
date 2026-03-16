"""
View TWOWAY motifs in the 16-18 residue range that would be gained
by expanding the max residue cutoff from 15 to 18.

Loads all isolatable, seq-deduped, A/C-filtered motifs with 16-18 residues,
groups them by topology, aligns within each topology group, and generates
a PyMOL session with all motifs visible.

Usage:
    python scripts/view_new_large_twoway_motifs.py -o large_twoway_view

    # Then open in PyMOL:
    pymol large_twoway_view/view.pml
"""

import copy
import os
import sys

import click
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_twoway_motifs import (
    get_cross_strand_bp_info,
    has_ac_internal,
    load_resolution_data,
)

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.tranforms import kabsch_algorithm

log = get_logger("view_large_twoway")

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


def get_new_large_motifs():
    """Load and filter to get the 16-18 residue TWOWAY motifs."""
    path = os.path.join(
        DATA_PATH, "summaries", "motifs", "non_redundant_motifs_summary.json"
    )
    print(f"Loading {path}")
    df_all = pd.read_json(path)
    df_tw = df_all[df_all["motif_type"] == "TWOWAY"].copy()
    print(f"Total TWOWAY motifs: {len(df_tw)}")

    # Resolution
    resolution_data = load_resolution_data()
    df_tw["resolution"] = df_tw["pdb_id"].map(resolution_data).fillna(99.0)

    # Isolatable filter
    df_tw = df_tw[df_tw["is_isolatable"] == 1].copy()
    print(f"After isolatable filter: {len(df_tw)}")

    # 16-18 residues only
    df_tw = df_tw[(df_tw["num_residues"] >= 16) & (df_tw["num_residues"] <= 18)].copy()
    print(f"With 16-18 residues: {len(df_tw)}")

    # Sequence dedup
    df_tw = df_tw.sort_values(["resolution", "motif_id"])
    df_tw = df_tw.groupby("motif_sequence", sort=False).first().reset_index()
    print(f"After sequence dedup: {len(df_tw)}")

    # A/C internal filter
    mask = df_tw["motif_sequence"].apply(has_ac_internal)
    df_tw = df_tw[mask].copy()
    print(f"After A/C internal filter: {len(df_tw)}")

    # Compute bp_sig
    bp_info = df_tw.apply(
        lambda row: get_cross_strand_bp_info(row, min_score=0.5), axis=1
    )
    df_tw["bp_sig"] = bp_info.apply(lambda x: x[0])
    df_tw["basepairs"] = bp_info.apply(lambda x: x[1])

    # Exclude very asymmetric topologies
    exclude_topos = {"10-2", "11-3", "12-2"}
    df_tw = df_tw[~df_tw["motif_topology"].isin(exclude_topos)].copy()
    print(f"After excluding topologies {exclude_topos}: {len(df_tw)}")

    # Exclude motifs with poor geometry
    exclude_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resources", "exclude_list", "excluded_twoway_motifs.csv",
    )
    if os.path.exists(exclude_csv):
        exclude_df = pd.read_csv(exclude_csv)
        exclude_ids = set(exclude_df["motif_id"])
        before = len(df_tw)
        df_tw = df_tw[~df_tw["motif_id"].isin(exclude_ids)].copy()
        print(f"After poor geometry exclude list: {len(df_tw)} (removed {before - len(df_tw)})")

    df_tw = df_tw.sort_values(["motif_topology", "bp_sig", "resolution"])
    return df_tw


@click.command()
@click.option(
    "-o", "--output", "output_dir", required=True,
    help="Output directory for CIF files and PyMOL script",
)
def main(output_dir):
    """View all new TWOWAY motifs in the 16-18 residue range."""
    df = get_new_large_motifs()
    print(f"\n{len(df)} motifs to visualize")

    os.makedirs(output_dir, exist_ok=True)
    motif_cache = {}
    abs_output_dir = os.path.abspath(output_dir)

    # Group by topology for alignment
    topo_groups = df.groupby("motif_topology")

    pml_lines = []
    pml_lines.append("# Auto-generated PyMOL script: new TWOWAY motifs (16-18 residues)")
    pml_lines.append(f"# Total motifs: {len(df)}")
    pml_lines.append("")
    pml_lines.append(f"cd {abs_output_dir}")
    pml_lines.append("")

    color_idx = 0
    total_exported = 0

    for topo, group in topo_groups:
        group = group.sort_values("resolution")
        n_motifs = len(group)
        has_bp_count = (group["bp_sig"] != "").sum()
        no_bp_count = (group["bp_sig"] == "").sum()

        print(f"\nTopology {topo}: {n_motifs} motif(s) "
              f"({has_bp_count} with bp, {no_bp_count} without)")

        pml_lines.append(f"# === Topology {topo} ({n_motifs} motifs) ===")

        # Load first motif as reference for alignment
        ref_row = group.iloc[0]
        ref_motif = load_motif(ref_row["motif_id"], ref_row["pdb_id"], motif_cache)
        if ref_motif is None:
            print(f"  Could not load reference motif {ref_row['motif_id']}")
            continue

        topo_member_names = []

        for i, (_, row) in enumerate(group.iterrows()):
            motif_id = row["motif_id"]
            pdb_id = row["pdb_id"]
            bp_sig = row["bp_sig"] if row["bp_sig"] else "(none)"
            seq = row["motif_sequence"]
            res = row["resolution"]
            bps = row.get("basepairs", [])

            bp_detail = ""
            if isinstance(bps, list) and bps:
                bp_detail = " ".join(f"{bp[0]}:{bp[1]}" for bp in bps)

            motif = load_motif(motif_id, pdb_id, motif_cache)
            if motif is None:
                print(f"  Could not load {motif_id}")
                continue

            if i == 0:
                aligned = motif
            else:
                aligned = align_motif_to_target(motif, ref_motif)
                if aligned is None:
                    # Different number of residues, just use unaligned
                    aligned = motif

            name = safe_name(motif_id)
            cif_path = os.path.join(output_dir, f"{name}.cif")
            aligned.to_cif(cif_path)

            color = COLORS[color_idx % len(COLORS)]
            label = (f"seq={seq}  bp_sig={bp_sig}  res={res:.2f}"
                     f"  nres={row['num_residues']}")
            if bp_detail:
                label += f"  bps={bp_detail}"

            pml_lines.append(f"# {label}")
            pml_lines.append(f"load {name}.cif, {name}")
            pml_lines.append(f"color {color}, {name}")

            topo_member_names.append(name)
            color_idx += 1
            total_exported += 1

            print(f"  {motif_id}  seq={seq}  bp_sig={bp_sig}  res={res:.2f}")

        # Group by topology
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
    for topo, _ in topo_groups:
        group_name = f"topo_{topo.replace('-', 'x')}"
        pml_lines.append(f"#   disable {group_name}")

    pml_path = os.path.join(output_dir, "view.pml")
    with open(pml_path, "w") as f:
        f.write("\n".join(pml_lines) + "\n")

    print(f"\nExported {total_exported} CIF files to {output_dir}/")
    print(f"PyMOL script: {pml_path}")
    print(f"\nOpen in PyMOL: pymol {pml_path}")


if __name__ == "__main__":
    main()
