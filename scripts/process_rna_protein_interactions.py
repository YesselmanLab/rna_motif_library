import os
import json
import glob
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict
from scipy.stats import entropy


from rna_motif_library.util import (
    get_pdb_ids,
    get_non_redundant_sets,
    get_cif_header_str,
    parse_residue_identifier,
    add_motif_indentifier_columns,
)
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.residue import get_cached_residues, Residue
from rna_motif_library.settings import DATA_PATH


def generate_res_motif_mapping(motifs):
    res_to_motif_id = {}
    for m in motifs:
        for r in m.get_residues():
            if r.get_str() not in res_to_motif_id:
                res_to_motif_id[r.get_str()] = m.name
            else:
                existing_motif = res_to_motif_id[r.get_str()]
                if existing_motif.startswith("HELIX"):
                    res_to_motif_id[r.get_str()] = m.name
    return res_to_motif_id


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


def process_pdb_hbonds(pdb_id):
    """Process hydrogen bonds for a single PDB ID.

    Args:
        pdb_id: The PDB ID to process
        all_unique_residues: Dictionary mapping PDB IDs to their unique residues

    Returns:
        DataFrame containing the processed hydrogen bonds or None if processing fails
    """
    path = os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_id}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None
    path = os.path.join(DATA_PATH, "dataframes", "protein_hbonds", f"{pdb_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    df = df[df["res_type_2"] == "PROTEIN"]
    df = df.reset_index(drop=True)
    df.drop(columns=["res_type_1", "res_type_2"], inplace=True)
    df["score"] = df["score"].round(3)
    df.to_csv(path, index=False)
    return df


def process_interaction_group(args):
    """Process a single group of RNA-protein interactions.

    Args:
        args: Tuple containing (group_key, group_df) where group_key is (res_type_1, res_type_2, atom_1, atom_2)
             and group_df is the DataFrame containing the interactions

    Returns:
        List containing the processed data for this group or None if group is too small
    """
    (res_1, res_2, atom_1, atom_2), g = args
    hist, x_edges, y_edges = make_2d_histogram(g, "angle_1", "dihedral_angle")
    if len(g) < 100:
        return None
    mean_score = g["score"].mean()
    mean_dihedral_angle = g["dihedral_angle"].mean()
    mean_angle_1 = g["angle_1"].mean()
    mean_angle_2 = g["angle_2"].mean()
    mean_distance = g["distance"].mean()
    return [
        res_1,
        res_2,
        atom_1,
        atom_2,
        len(g),
        normalized_gini_by_datapoints(hist),
        mean_score,
        mean_dihedral_angle,
        mean_angle_1,
        mean_angle_2,
        mean_distance,
    ]


def process_pdb_motifs(pdb_id):
    """Process motifs and hydrogen bonds for a single PDB ID.

    Args:
        args: Tuple containing (pdb_id, group_df) where group_df contains the hydrogen bonds for this PDB ID

    Returns:
        DataFrame containing the motif hydrogen bonds or None if no motifs found
    """
    try:
        df = pd.read_csv(
            os.path.join(DATA_PATH, "dataframes", "protein_hbonds", f"{pdb_id}.csv")
        )
    except:
        return None
    path = os.path.join(
        DATA_PATH, "dataframes", "motif_protein_interactions", f"{pdb_id}.json"
    )
    if os.path.exists(path):
        return pd.read_json(path)

    motifs = get_cached_motifs(pdb_id)
    motif_hbond_dict = defaultdict(list)
    res_to_motif_id = generate_res_motif_mapping(motifs)
    for i, row in df.iterrows():
        if row["res_1"] not in res_to_motif_id:
            continue
        motif_name = res_to_motif_id[row["res_1"]]
        motif_hbond_dict[motif_name].append(
            [
                row["res_1"],
                row["atom_1"],
                row["res_2"],
                row["atom_2"],
                row["score"],
            ]
        )

    df = pd.DataFrame(motif_hbond_dict.items(), columns=["motif_id", "hbonds"])
    if len(df) == 0:
        return None
    df["num_hbonds"] = df["hbonds"].apply(len)
    keep = [True] * len(df)
    num_protein_chains = []
    num_unique_res_1 = []
    num_unique_res_2 = []
    for i, row in df.iterrows():
        unique_res_1 = []
        unique_res_2 = []
        unique_protein_chains = []
        for hbond in row["hbonds"]:
            if hbond[0] not in unique_res_1:
                unique_res_1.append(hbond[0])
            if hbond[2] not in unique_res_2:
                unique_res_2.append(hbond[2])
            res_info = parse_residue_identifier(hbond[2])
            if res_info["chain_id"] not in unique_protein_chains:
                unique_protein_chains.append(res_info["chain_id"])
        num_protein_chains.append(len(unique_protein_chains))
        num_unique_res_1.append(len(unique_res_1))
        num_unique_res_2.append(len(unique_res_2))
    df["num_protein_chains"] = num_protein_chains
    df["num_rna_res"] = num_unique_res_1
    df["num_protein_res"] = num_unique_res_2
    df = df[keep]
    df = df.sort_values(by="num_hbonds", ascending=False)
    df = add_motif_indentifier_columns(df, "motif_id")
    df.to_json(path, orient="records")
    return df


# cli #################################################################################


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def get_rna_prot_hbonds(processes):
    pdb_ids = get_pdb_ids()
    # Process PDB IDs in parallel
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "protein_hbonds"), exist_ok=True)
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_hbonds,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for RNA-protein hydrogen bonds",
    )


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def analyze_rna_prot_hbonds(processes):
    df = pd.read_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "protein_interactions",
            "non_redundant_protein_hbonds.json",
        )
    )
    df["res_type_1"] = df["res_1"].str.split("-").str[1]
    df["res_type_2"] = df["res_2"].str.split("-").str[1]

    df_struct = pd.read_csv("data/csvs/rna_structures.csv")
    df_struct.rename(columns={"pdb_id": "pdb_name"}, inplace=True)
    df = df.merge(df_struct, on="pdb_name", how="left")

    # Create list of groups to process
    groups = list(df.groupby(["res_type_1", "res_type_2", "atom_1", "atom_2"]))

    # Process groups in parallel
    results = run_w_processes_in_batches(
        items=groups,
        func=process_interaction_group,
        processes=processes,
        batch_size=50,
        desc="Processing RNA-protein interaction groups",
    )

    # Filter out None results and create DataFrame
    data = [r for r in results if r is not None]
    if not data:
        print("No results found")
        return

    df = pd.DataFrame(
        data,
        columns=[
            "res_1",
            "res_2",
            "atom_1",
            "atom_2",
            "num_datapoints",
            "normalized_gini",
            "mean_score",
            "mean_dihedral_angle",
            "mean_angle_1",
            "mean_angle_2",
            "mean_distance",
        ],
    )
    df.sort_values(by="normalized_gini", ascending=False, inplace=True)
    df.to_csv("rna_protein_hbonds_normalized_gini.csv", index=False)


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def motif_analysis(processes):
    # Process PDB IDs in parallel
    pdb_ids = get_pdb_ids()
    os.makedirs(
        os.path.join(DATA_PATH, "dataframes", "motif_protein_interactions"),
        exist_ok=True,
    )

    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_motifs,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for motif analysis",
    )


@cli.command()
def concat_rna_protein_hbonds():
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "protein_hbonds", "*.csv")
    )
    df = concat_dataframes_from_files(csv_files)
    print(len(df))
    path = os.path.join(
        DATA_PATH, "summaries", "protein_interactions", "all_protein_hbonds.csv"
    )
    df.to_csv(path, index=False)
    unique_res = json.load(
        open(os.path.join(DATA_PATH, "summaries", "unique_residues.json"))
    )
    unique_res_dict = {}
    for d in unique_res:
        unique_res_dict[d["pdb_id"]] = d["residues"]
    keep = [True] * len(df)
    for i, row in df.iterrows():
        if row["pdb_name"] not in unique_res_dict:
            keep[i] = False
            continue
        if row["res_1"] not in unique_res_dict[row["pdb_name"]]:
            keep[i] = False
    df = df[keep]
    df.to_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "protein_interactions",
            "non_redundant_protein_hbonds.json",
        ),
        orient="records",
    )


@cli.command()
def concat_motif_analysis():
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "motif_protein_interactions", "*.json")
    )
    df = concat_dataframes_from_files(json_files)
    df_unique_motifs = pd.read_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    )
    pdb_ids = df_unique_motifs["pdb_id"].values
    unique_motifs = list(df_unique_motifs["motif_name"].values)
    print(len(df))
    path = os.path.join(
        DATA_PATH,
        "summaries",
        "protein_interactions",
        "all_motif_protein_interactions.json",
    )
    df.to_json(path, orient="records")
    df = df.query("motif_id in @unique_motifs")
    df.to_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "protein_interactions",
            "non_redundant_motif_protein_interactions.json",
        ),
        orient="records",
    )


@cli.command()
@click.argument("res_1")
@click.argument("res_2")
@click.argument("atom_1")
@click.argument("atom_2")
def plot_histogram(res_1, res_2, atom_1, atom_2):
    df = pd.read_csv(f"data/protein_interactions/{res_1}_{res_2}.csv")
    df = df.query(f"atom_1 == '{atom_1}' and atom_2 == '{atom_2}'")
    plt.hist2d(df["angle_1"], df["dihedral_angle"], bins=30, cmap="viridis")
    plt.colorbar(label="Frequency")
    plt.xlabel("Angle 1")
    plt.ylabel("Dihedral Angle")
    plt.title(f"{res_1} {atom_1} - {res_2} {atom_2}")
    plt.show()


@cli.command()
def plot_histograms():
    df = pd.read_csv("rna_protein_hbonds.csv")
    df_scores = pd.read_csv("rna_protein_hbonds_normalized_gini.csv")
    scores = {}
    for i, row in df_scores.iterrows():
        key = (
            row["res_1"]
            + "_"
            + row["res_2"]
            + "_"
            + row["atom_1"]
            + "_"
            + row["atom_2"]
        )
        scores[key] = [row["normalized_gini"], row["mean_score"]]
    df["res_type_1"] = df["res_1"].str.split("-").str[1]
    df["res_type_2"] = df["res_2"].str.split("-").str[1]
    for i, g in df.groupby(["res_type_1", "res_type_2", "atom_1", "atom_2"]):
        res_1, res_2, atom_1, atom_2 = i[0], i[1], i[2], i[3]
        if len(g) < 100:
            continue
        print(res_1, res_2, atom_1, atom_2)
        key = res_1 + "_" + res_2 + "_" + atom_1 + "_" + atom_2
        if key not in scores:
            continue
        normalized_gini, mean_score = scores[key]
        normalized_gini = round(normalized_gini, 3)
        if normalized_gini < 0.6:
            continue
        mean_score = round(mean_score, 3)
        plt.hist2d(g["angle_1"], g["dihedral_angle"], bins=30, cmap="Greys")
        plt.colorbar(label="Frequency")
        plt.xlabel("Angle")
        plt.ylabel("Dihedral Angle")
        plt.title(f"{res_1} {atom_1} - {res_2} {atom_2}")
        plt.savefig(
            f"plots/rna_protein_histograms/{res_1}_{res_2}_{atom_1}_{atom_2}_{normalized_gini}_{mean_score}.png",
            dpi=300,
        )
        plt.close()
        plt.clf()  # Clear the current figure


@cli.command()
@click.argument("res_1")
@click.argument("res_2")
@click.argument("atom_1")
@click.argument("atom_2")
def get_structures(res_1, res_2, atom_1, atom_2):
    df = pd.read_csv(f"data/protein_interactions/{res_1}_{res_2}.csv")
    df = df.query(f"atom_1 == '{atom_1}' and atom_2 == '{atom_2}'")
    count = 0
    for i, g in df.groupby("pdb_name"):
        residues = get_cached_residues(i)
        for j, row in g.iterrows():
            res_1 = residues[row["res_1"]]
            res_2 = residues[row["res_2"]]
            write_residues_to_cif([res_1, res_2], "test.cif")
            exit()


if __name__ == "__main__":
    cli()
