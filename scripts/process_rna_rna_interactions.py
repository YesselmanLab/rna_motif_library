import os
import json
import glob
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import entropy


from rna_motif_library.util import (
    get_pdb_ids,
    get_non_redundant_sets,
    get_cif_header_str,
)
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.residue import get_cached_residues, Residue
from rna_motif_library.settings import DATA_PATH


def write_residues_to_cif(residues: List[Residue], output_path: str) -> None:
    """
    Write a list of residues to a mmCIF format file.

    Args:
        residues: List of Residue objects to write
        output_path: Path to output .cif file
    """
    with open(output_path, "w") as f:
        # Write header
        f.write(get_cif_header_str())

        # Write each residue
        acount = 1
        for res in residues:
            f.write(res.to_cif_str(acount))
            acount += len(res.coords)


def get_unique_res(pdb_id, motifs):
    path = os.path.join(DATA_PATH, "dataframes", "duplicate_motifs", f"{pdb_id}.csv")
    dup_motifs = []
    unique_res = []
    if os.path.exists(path):
        try:
            df_dup = pd.read_csv(path)
            dup_motifs = df_dup["dup_motif"].values
        except Exception as e:
            dup_motifs = []

    for m in motifs:
        if m.name in dup_motifs:
            continue
        for r in m.get_residues():
            if r.get_str() not in unique_res:
                unique_res.append(r.get_str())
    return unique_res


def make_2d_histogram(df, col1, col2, bins=30):
    """
    Creates a 2D histogram comparing values from two columns in a dataframe

    Args:
        df (pd.DataFrame): Input dataframe
        col1 (str): First column name
        col2 (str): Second column name
        bins (int): Number of bins for histogram (default 30)

    Returns:
        tuple: (histogram array, x edges array, y edges array)
    """
    hist, x_edges, y_edges = np.histogram2d(np.abs(df[col1]), df[col2], bins=bins)
    return hist, x_edges, y_edges


def gini_coefficient(hist2d):
    hist_flat = hist2d.flatten()
    sorted_hist = np.sort(hist_flat)
    n = len(sorted_hist)
    cumulative = np.cumsum(sorted_hist)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def gini_coefficient_from_counts(counts):
    """Compute the Gini coefficient given an array of counts.
    Counts should be a 1D numpy array.
    """
    if counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    # Compute Gini: maximum inequality is when all mass is in one bin.
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def expected_gini_uniform(m, n, num_simulations=500):
    """Estimate the expected Gini coefficient under uniform distribution
    given m total points and n bins.
    """
    ginis = []
    for _ in range(num_simulations):
        # simulate m data points uniformly distributed among n bins
        counts = np.random.multinomial(m, [1 / n] * n)
        ginis.append(gini_coefficient_from_counts(counts))
    return np.mean(ginis)


def normalized_gini_by_datapoints(hist2d, num_simulations=500):
    """
    Computes an adjusted Gini coefficient for a 2D histogram,
    taking into account the number of data points.

    The histogram is first flattened to a 1D array of counts.
    """
    counts = hist2d.flatten()
    m = counts.sum()  # total number of data points
    n = len(counts)

    # Observed Gini from the counts
    g_obs = gini_coefficient_from_counts(counts)

    # Expected Gini for m data points uniformly distributed across n bins.
    g_expected = expected_gini_uniform(m, n, num_simulations)

    # Maximum Gini for this number of bins (all mass in one bin)
    g_max = (n - 1) / n

    # Adjusted Gini: 0 if the observed is at the uniform baseline,
    # and 1 if it is as concentrated as possible.
    if g_max - g_expected == 0:
        return 0.0
    return (g_obs - g_expected) / (g_max - g_expected)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output", type=str, default="rna_rna_interactions.csv")
def get_rna_rna_hbonds(output):
    all_unique_residues = json.load(
        open(os.path.join(DATA_PATH, "summaries", "unique_residues.json"))
    )
    all_unique_residues = {d["pdb_id"]: d["residues"] for d in all_unique_residues}
    count = 0
    pdb_ids = get_pdb_ids()
    dfs = []
    for pdb_id in pdb_ids:
        if pdb_id not in all_unique_residues:
            continue
        unique_res = all_unique_residues[pdb_id]
        try:
            motifs = get_cached_motifs(pdb_id)
        except Exception as e:
            continue
        helix_hbonds = {}
        for m in motifs:
            res = []
            for r in m.get_residues():
                res.append(r.get_str())
            if m.mtype != "HELIX":
                continue
            for bp in m.basepairs:
                if bp.lw != "cWW":
                    continue
                if bp.res_1.get_str() in res and bp.res_2.get_str() in res:
                    helix_hbonds[bp.res_1.get_str() + "-" + bp.res_2.get_str()] = 1
        path = os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_id}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            continue
        df = df[df["res_type_2"] == "RNA"]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            continue
        has_uniq_res = [False] * len(df)
        for i, row in df.iterrows():
            key = row["res_1"] + "-" + row["res_2"]
            key_rev = row["res_2"] + "-" + row["res_1"]
            if key in helix_hbonds or key_rev in helix_hbonds:
                continue
            if row["res_1"] in unique_res:
                has_uniq_res[i] = True
        df = df[has_uniq_res]
        count += len(df)
        print(pdb_id, len(df), count)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(output, index=False)


@cli.command()
def split_rna_rna_hbonds():
    df = pd.read_csv("rna_rna_interactions.csv")
    df["res_1_type"] = df["res_1"].str.split("-").str[1]
    df["res_2_type"] = df["res_2"].str.split("-").str[1]
    for i, g in df.groupby(["res_1_type", "res_2_type"]):
        print(i, len(g))
        name = f"{i[0]}_{i[1]}.csv"
        g.to_csv(f"data/rna_interactions/{name}", index=False)


if __name__ == "__main__":
    cli()
