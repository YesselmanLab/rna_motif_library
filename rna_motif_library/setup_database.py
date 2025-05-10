import re
import requests
import wget
import os
import glob
import pandas as pd
import click
import threading
import concurrent.futures

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.logger import get_logger, setup_logging
from rna_motif_library.pdb_queries import get_rna_structures

log = get_logger("setup-database")

# helper functions ###################################################################


def get_pdb_ids_from_non_redundant_set(csv_path):
    df = pd.read_csv(csv_path, header=None)
    data = []
    for _, row in df.iterrows():
        row = row.tolist()
        spl = row[1].split("|")
        if spl[0] not in data:
            data.append(spl[0])
    df = pd.DataFrame(data, columns=["pdb_id"])
    return df


def download_cif(pdb_id):
    """
    Downloads a PDB file based on the given row.

    Args:
        pdb_id (str): The PDB ID of the structure to download.

    Returns:
        None: If the file is already downloaded, the function returns None.

    Raises:
        Exception: If there is an error while downloading the PDB file.

    """
    out_path = os.path.join(DATA_PATH, "pdbs", f"{pdb_id}.cif")
    if os.path.isfile(out_path):
        return  # Skip this row because the file is already downloaded
    try:
        log.info(f"Downloading {pdb_id}")
        wget.download(f"https://files.rcsb.org/download/{pdb_id}.cif", out=out_path)
    except Exception as e:
        log.error(f"Failed to download {pdb_id}: {e}")


# cli #################################################################################


@click.group()
def cli():
    pass


# STEP 1: Get the latest RNA 3D Hub releases
@cli.command()
def get_atlas_release():
    """
    Fetches the latest RNA 3D Hub release number from the current release page and downloads
    the corresponding non-redundant RNA structure dataset as a CSV file.

    The function:
    1. Gets the current release number from RNA 3D Hub
    2. Downloads the non-redundant set CSV file for structures filtered at 3.5Å resolution
    3. Saves the CSV file to the atlas directory under DATA_PATH/csvs/
    """
    setup_logging()
    url = "https://rna.bgsu.edu/rna3dhub/nrlist/release/current"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None

    # Search for the release number pattern in the page content
    match = re.search(r"Release\s+(\d+\.\d+)", response.text)
    if match:
        release_number = match.group(1)
        log.info(f"Latest RNA 3D Hub release: {release_number}")
    else:
        log.error("Release number not found in the page content.")
        exit(1)
    atlas_csv_path = os.path.join(DATA_PATH, "csvs", "atlas")
    wget.download(
        f"https://rna.bgsu.edu/rna3dhub/nrlist/download/{release_number}/3.5A/csv",
        os.path.join(atlas_csv_path, f"non_redundant_set_{release_number}.csv"),
    )
    wget.download(
        f"https://rna.bgsu.edu/rna3dhub/motifs/release/hl/current/csv",
        atlas_csv_path,
    )
    wget.download(
        f"https://rna.bgsu.edu/rna3dhub/motifs/release/il/current/csv",
        atlas_csv_path,
    )
    wget.download(
        f"https://rna.bgsu.edu/rna3dhub/motifs/release/J3/current/csv",
        atlas_csv_path,
    )


# STEP 2: Get all RNA PDBs with resolution better than 3.5Å
@cli.command()
def get_all_rna_pdbs():
    """
    Fetches all RNA structures from RCSB PDB with resolution better than 3.5Å.

    The function:
    1. Queries RCSB PDB API for RNA structures with resolution better than 3.5Å
    2. Saves the PDB IDs to a text file
    """
    setup_logging()
    atlas_csv_path = os.path.join(DATA_PATH, "csvs", "atlas", "non_redundant_set_*.csv")
    atlas_csv_files = glob.glob(atlas_csv_path)
    if len(atlas_csv_files) == 0:
        log.error("No atlas CSV files found in the atlas directory.")
        exit(1)
    df_atlas_pdbs = get_pdb_ids_from_non_redundant_set(atlas_csv_files[0])
    rna_pdbs = get_rna_structures(3.51)
    df_rna_pdbs = pd.DataFrame(rna_pdbs, columns=["pdb_id", "resolution"])
    # Find PDBs in atlas that are not in rna_pdbs
    atlas_only = df_atlas_pdbs[~df_atlas_pdbs["pdb_id"].isin(df_rna_pdbs["pdb_id"])]
    if len(atlas_only) > 0:
        log.info(f"Found {len(atlas_only)} PDBs in atlas that are not in RNA PDB set:")
        for pdb in atlas_only["pdb_id"]:
            log.info(pdb)
    # Merge the dataframes, keeping all PDBs from both sets
    df_merged = pd.merge(df_rna_pdbs, df_atlas_pdbs, on="pdb_id", how="outer")
    # Save to CSV file in the data directory
    output_path = os.path.join(DATA_PATH, "csvs", "rna_structures.csv")
    df_merged.to_csv(output_path, index=False)
    log.info(f"Saved {len(df_merged)} RNA structures to {output_path}")


# OPTIONAL STEP: generate splits so can be run in parallel
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("n_splits", type=int)
def generate_splits(csv_path, n_splits):
    os.makedirs("splits", exist_ok=True)
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    # Create n_splits CSV files containing evenly distributed PDB IDs
    split_size = len(pdb_ids) // n_splits
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_splits - 1 else len(pdb_ids)
        split_pdb_ids = pdb_ids[start_idx:end_idx]

        df = pd.DataFrame({"pdb_id": split_pdb_ids})
        df.to_csv(f"splits/split_{i}.csv", index=False)


# STEP 3: download all RNA cif files
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(csv_path, threads):
    """
    Downloads all RNA PDBs from the RNA structures CSV file.
    """
    setup_logging()
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_cif, pdb_ids)


if __name__ == "__main__":
    cli()
