import pandas as pd
import click
import glob
import os
import json

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.parallel_utils import concat_dataframes_from_files, run_w_processes_in_batches
from rna_motif_library.util import add_motif_indentifier_columns


def process_pdb_motifs(args):
    pdb_id, df = args
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {m.name: m for m in motifs}
    residues_list = []
    for i, row in df.iterrows():
        m = motifs_by_name[row["motif_id"]]
        residues = []
        for r in m.get_residues():
            residues.append(r.get_str())
        residues_list.append(residues)
    df["residues"] = residues_list
    return df


# cli commands ########################################################################

@click.group()
def cli():
    """
    main function for script
    """


# TODO need to update this
@cli.command()
def get_pdb_info():
    RELEASE_PATH = os.path.join("release", "pdb_info")
    df = pd.read_csv("data/csvs/rna_structures.csv")
    df_count = pd.read_csv("data/csvs/rna_residue_counts.csv")
    df_count = df_count.rename(columns={"count": "num_rna_residues"})
    df = df.merge(df_count, on="pdb_id", how="left")
    df_pdb_titles = pd.read_json("data/summaries/pdb_titles.json")
    df_pdb_titles["pdb_id"] = df_pdb_titles["pdb_id"].str.upper()
    df = df.merge(df_pdb_titles, on="pdb_id", how="left")
    df.to_csv(os.path.join(RELEASE_PATH, "pdb_info.csv"), index=False)


@cli.command()
def get_hbonds():
    RELEASE_PATH = os.path.join("release", "hbonds")
    all_unique_residues = json.load(
        open(os.path.join(DATA_PATH, "summaries", "unique_residues.json"))
    )
    all_unique_residues = {d["pdb_id"]: d["residues"] for d in all_unique_residues}
    df = concat_dataframes_from_files(
        glob.glob(os.path.join(DATA_PATH, "dataframes", "hbonds", "*.csv"))
    )
    df.rename(
        columns={
            "res_1": "res_id_1",
            "res_2": "res_id_2",
            "atom_1": "atom_name_1",
            "atom_2": "atom_name_2",
            "score": "hbond_score",
            "pdb_name": "pdb_id",
        },
        inplace=True,
    )
    df["hbond_score"] = df["hbond_score"].round(3)
    df.to_csv(os.path.join(RELEASE_PATH, "all_hbonds.csv"), index=False)
    os.system("gzip -9 -f {}".format(os.path.join(RELEASE_PATH, "all_hbonds.csv")))
    has_uniq_res = [False] * len(df)
    for i, row in df.iterrows():
        if row["pdb_id"] not in all_unique_residues:
            continue
        if (
            row["res_id_1"] in all_unique_residues[row["pdb_id"]]
            and row["res_id_2"] in all_unique_residues[row["pdb_id"]]
        ):
            has_uniq_res[i] = True
    df = df[has_uniq_res]
    df.to_csv(os.path.join(RELEASE_PATH, "non_redundant_hbonds.csv"), index=False)
    os.system(
        "gzip -9 -f {}".format(os.path.join(RELEASE_PATH, "non_redundant_hbonds.csv"))
    )


@cli.command()
def get_basepairs():
    RELEASE_PATH = os.path.join("release", "basepairs")
    all_unique_residues = json.load(
        open(os.path.join(DATA_PATH, "summaries", "unique_residues.json"))
    )
    all_unique_residues = {d["pdb_id"]: d["residues"] for d in all_unique_residues}
    df = concat_dataframes_from_files(
        glob.glob(os.path.join(DATA_PATH, "dataframes", "basepairs", "*.json"))
    )
    df.rename(
        columns={
            "res_1": "res_id_1",
            "res_2": "res_id_2",
        },
        inplace=True,
    )
    df["hbond_score"] = df["hbond_score"].round(3)
    df_unique = df.copy()
    df = df.drop(columns=["ref_frame"])
    df.to_csv(os.path.join(RELEASE_PATH, "all_basepairs.csv"), index=False)
    os.system("gzip -9 -f {}".format(os.path.join(RELEASE_PATH, "all_basepairs.csv")))
    has_uniq_res = [False] * len(df_unique)
    for i, row in df_unique.iterrows():
        if row["pdb_id"] not in all_unique_residues:
            continue
        if (
            row["res_id_1"] in all_unique_residues[row["pdb_id"]]
            and row["res_id_2"] in all_unique_residues[row["pdb_id"]]
        ):
            has_uniq_res[i] = True
    df_unique = df_unique[has_uniq_res]
    df_unique.to_json(
        os.path.join(RELEASE_PATH, "non_redundant_basepairs.json"), orient="records"
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(RELEASE_PATH, "non_redundant_basepairs.json")
        )
    )

@cli.command()
@click.option("-p", "--processes", type=int, default=1, help="Number of processes to use")
def get_motifs(processes):
    RELEASE_PATH = os.path.join("release", "motifs")
    df = pd.read_csv(os.path.join(DATA_PATH, "summaries", "all_motifs.csv"))
    df = add_motif_indentifier_columns(df, "motif_id")
    df.rename(
        columns={
            "mtype": "motif_type",
            "size": "motif_topology",
            "sequence": "motif_sequence",
        },
        inplace=True,
    )
    df["residues"] = [[] for _ in range(len(df))]
    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=list(df.groupby("pdb_id")),
        func=process_pdb_motifs,
        processes=processes,
        batch_size=100,
        desc="Processing motifs",
    )
    # Combine results
    df = pd.concat(results)
    df.to_json(os.path.join(RELEASE_PATH, "all_motifs.json"), orient="records")
    os.system("gzip -9 -f {}".format(os.path.join(RELEASE_PATH, "all_motifs.json")))


@cli.command()
def get_tertiary_contacts():
    RELEASE_PATH = os.path.join("release", "tertiary_contacts")
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", "*.json")
    )
    df = concat_dataframes_from_files(json_files)
    # df = pd.read_json("data/dataframes/tertiary_contacts/1GID.json")
    df.rename(
        columns={
            "motif_1": "motif_id_1",
            "motif_2": "motif_id_2",
            "mtype_1": "motif_type_1",
            "mtype_2": "motif_type_2",
            "motif_1_res": "motif_1_interacting_residues",
            "motif_2_res": "motif_2_interacting_residues",
            "base-base": "num_base_base_hbonds",
            "base-sugar": "num_base_sugar_hbonds",
            "base-phos": "num_base_phosphate_hbonds",
            "phos-sugar": "num_phosphate_sugar_hbonds",
            "phos-phos": "num_phosphate_phosphate_hbonds",
            "unique_motif_1": "is_motif_1_unique",
            "unique_motif_2": "is_motif_2_unique",
        },
        inplace=True,
    )
    df["is_motif_1_unique"] = df["is_motif_1_unique"].astype(int)
    df["is_motif_2_unique"] = df["is_motif_2_unique"].astype(int)
    df.to_json(
        os.path.join(RELEASE_PATH, "all_tertiary_contacts.json"), orient="records"
    )
    os.system(
        "gzip -9 -f {}".format(os.path.join(RELEASE_PATH, "all_tertiary_contacts.json"))
    )
    df = df[~((df["is_motif_1_unique"] == 0) & (df["is_motif_2_unique"] == 0))]
    df.to_json(
        os.path.join(RELEASE_PATH, "unique_tertiary_contacts.json"), orient="records"
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(RELEASE_PATH, "unique_tertiary_contacts.json")
        )
    )


@cli.command()
def get_protein_interactions():
    RELEASE_PATH = os.path.join("release", "protein_interactions")
    df_unique = pd.read_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    )
    unique_motif_ids = df_unique["motif_name"].unique()
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "motif_protein_interactions", "*.json")
    )
    df = concat_dataframes_from_files(json_files)
    df.to_json(
        os.path.join(RELEASE_PATH, "all_protein_interactions.json"), orient="records"
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(RELEASE_PATH, "all_protein_interactions.json")
        )
    )
    df = df[df["motif_id"].isin(unique_motif_ids)]
    df.to_json(
        os.path.join(RELEASE_PATH, "non_redundant_protein_interactions.json"),
        orient="records",
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(RELEASE_PATH, "non_redundant_protein_interactions.json")
        )
    )


if __name__ == "__main__":
    cli()
