import re
import requests
import wget
import os
import glob
import pandas as pd
import click
import time
import json
import subprocess

from biopandas.mmcif import PandasMmcif

from rna_motif_library.chain import (
    get_rna_chains,
    get_protein_chains,
    save_chains_to_json,
)
from rna_motif_library.hbond import generate_hbonds, save_hbonds_to_json
from rna_motif_library.basepair import generate_basepairs, save_basepairs_to_json
from rna_motif_library.settings import DATA_PATH, DSSR_EXE
from rna_motif_library.logger import get_logger, setup_logging
from rna_motif_library.ligand import generate_ligand_info
from rna_motif_library.pdb_queries import get_rna_structures, get_pdb_titles_batch
from rna_motif_library.residue import (
    Residue,
    save_residues_to_json,
    get_cached_residues,
)
from rna_motif_library.util import (
    sanitize_x3dna_atom_name,
    get_cached_path,
)
from rna_motif_library.parallel_utils import run_w_processes_w_batches, run_w_threads
from rna_motif_library.x3dna import get_residue_type, get_cached_dssr_output
from rna_motif_library.motif import save_motifs_to_json
from rna_motif_library.motif_factory import MotifFactory, get_pdb_structure_data

log = get_logger("setup-database")

# helper functions ###################################################################


def get_pdb_ids_from_non_redundant_set(csv_path):
    """
    Extracts unique PDB IDs from the RNA 3D Hub non-redundant set CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the non-redundant set data.

    Returns:
        pandas.DataFrame: DataFrame containing unique PDB IDs with a single column 'pdb_id'.
    """
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


def process_cif(cif_file):
    cols = [
        "group_PDB",
        "id",
        "auth_atom_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]

    pdb_name = os.path.basename(cif_file).split(".")[0]
    out_path = os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_name}.parquet")
    if os.path.exists(out_path):
        return pdb_name
    try:
        log.info(f"Processing {pdb_name}")
        ppdb = PandasMmcif().read_mmcif(cif_file)
        df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
        df = df[cols]
        # Write with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df.to_parquet(f"data/pdbs_dfs/{pdb_name}.parquet")
                break
            except TimeoutError:
                if attempt == max_retries - 1:
                    print(f"Failed to write {pdb_name} after {max_retries} attempts")
                    raise
                print(f"Timeout writing {pdb_name}, retrying...")
                time.sleep(1)  # Wait before retry
        return pdb_name
    except Exception as e:
        print(f"Error processing {pdb_name}: {str(e)}")
        return None


def run_dssr_on_pdb(dssr_path, pdb_path, out_path):
    subprocess.run(
        f"{dssr_path} -i={pdb_path} -o={out_path} --json --more 2> /dev/null",
        shell=True,
    )
    files = glob.glob("dssr-*")
    for f in files:
        os.remove(f)


def generate_dssr_output(pdb_id):
    """
    Generates DSSR output for a given PDB ID.
    """
    if os.path.exists(get_cached_path(pdb_id, "dssr_output")):
        return pdb_id
    tries = 0
    while tries < 3:
        log.info(f"Generating DSSR output for {pdb_id}")
        run_dssr_on_pdb(
            DSSR_EXE,
            os.path.join(DATA_PATH, "pdbs", f"{pdb_id}.cif"),
            get_cached_path(pdb_id, "dssr_output"),
        )
        # sometimes data is not saved correctly, so we try 3 times
        try:
            data = json.load(open(get_cached_path(pdb_id, "dssr_output")))
            return pdb_id
        except Exception as e:
            log.error(f"Failed to generate DSSR output for {pdb_id}: {e}")
            tries += 1


def process_residues_in_pdb(pdb_id):
    """
    Processes residues from source PDB using data from DSSR.
    """
    if os.path.exists(get_cached_path(pdb_id, "residues")):
        return pdb_id
    df_atoms = pd.read_parquet(os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_id}.parquet"))
    log.info(f"Processing residues for {pdb_id}")
    residues = {}
    for i, g in df_atoms.groupby(
        ["auth_asym_id", "auth_seq_id", "auth_comp_id", "pdbx_PDB_ins_code"]
    ):
        coords = g[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        atom_names = g["auth_atom_id"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        chain_id, res_num, res_name, ins_code = i
        if ins_code == "None" or ins_code is None:
            ins_code = ""
        res = Residue(
            chain_id,
            res_name,
            int(res_num),
            ins_code,
            get_residue_type(res_name),
            atom_names,
            coords,
        )
        residues[res.get_str()] = res
    # Save residues to json file
    save_residues_to_json(residues, get_cached_path(pdb_id, "residues"))
    return pdb_id


def process_chains_in_pdb(pdb_id):
    """
    Processes chains from source PDB using data from DSSR.
    """
    if os.path.exists(get_cached_path(pdb_id, "chains")):
        return pdb_id
    print(f"Processing chains for {pdb_id}")
    residues = get_cached_residues(pdb_id)
    chains = get_rna_chains(list(residues.values()))
    save_chains_to_json(chains, get_cached_path(pdb_id, "chains"))
    chains = get_protein_chains(list(residues.values()))
    save_chains_to_json(chains, get_cached_path(pdb_id, "protein_chains"))
    return pdb_id


def process_interactions_in_pdb(pdb_id):
    """
    Processes hbonds and basepairs from source PDB.
    """
    if os.path.exists(get_cached_path(pdb_id, "hbonds")) and os.path.exists(
        get_cached_path(pdb_id, "basepairs")
    ):
        return pdb_id
    print(f"Processing interactions for {pdb_id}")
    residues = get_cached_residues(pdb_id)
    hbonds = generate_hbonds(pdb_id)
    residues = get_cached_residues(pdb_id)
    dssr_output = get_cached_dssr_output(pdb_id)
    dssr_pairs = dssr_output.get_pairs()
    basepairs = generate_basepairs(pdb_id, dssr_pairs, residues)
    log.info(
        f"Processed {pdb_id} with {len(hbonds)} hbonds and {len(basepairs)} basepairs"
    )
    save_hbonds_to_json(hbonds, get_cached_path(pdb_id, "hbonds"))
    save_basepairs_to_json(basepairs, get_cached_path(pdb_id, "basepairs"))
    return pdb_id


def generate_motifs_in_pdb(pdb_id):
    """
    Generates motifs for a given PDB ID.
    """
    if os.path.exists(get_cached_path(pdb_id, "motifs")):
        return pdb_id
    print(f"Generating motifs for {pdb_id}")
    pdb_data = get_pdb_structure_data(pdb_id)
    motif_factory = MotifFactory(pdb_data)
    motifs = motif_factory.get_motifs()
    save_motifs_to_json(motifs, get_cached_path(pdb_id, "motifs"))
    return pdb_id


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
    if not os.path.exists(os.path.join(DATA_PATH, "csvs", "atlas")):
        log.info(
            "Creating atlas directory: " + os.path.join(DATA_PATH, "csvs", "atlas")
        )
        os.makedirs(os.path.join(DATA_PATH, "csvs", "atlas"))
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
    setup_logging()
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()

    # Get file sizes for each PDB
    file_sizes = {}
    for pdb_id in pdb_ids:
        cif_path = os.path.join(DATA_PATH, "pdbs", f"{pdb_id}.cif")
        if os.path.exists(cif_path):
            file_sizes[pdb_id] = os.path.getsize(cif_path)
        else:
            # Assign a default size if file doesn't exist yet
            file_sizes[pdb_id] = 0

    # Sort PDBs by file size (largest first)
    sorted_pdb_ids = sorted(pdb_ids, key=lambda x: file_sizes.get(x, 0), reverse=True)

    # Initialize splits with empty lists
    splits = [[] for _ in range(n_splits)]
    split_sizes = [0] * n_splits

    # Distribute PDBs using a greedy approach (assign to the split with smallest total size)
    for pdb_id in sorted_pdb_ids:
        # Find the split with the smallest total size
        min_size_idx = split_sizes.index(min(split_sizes))
        splits[min_size_idx].append(pdb_id)
        split_sizes[min_size_idx] += file_sizes.get(pdb_id, 0)

    # Save each split to a CSV file
    for i, split_pdb_ids in enumerate(splits):
        df = pd.DataFrame({"pdb_id": split_pdb_ids})
        df.to_csv(f"splits/split_{i}.csv", index=False)
        log.info(
            f"Split {i}: {len(split_pdb_ids)} PDBs, total size: {split_sizes[i]/1024/1024:.2f} MB"
        )


# STEP 3: download all RNA cif files
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(csv_path, threads):
    """
    Downloads all RNA PDBs from the RNA structures CSV file.
    """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "pdbs")):
        log.info("Creating pdbs directory: " + os.path.join(DATA_PATH, "pdbs"))
        os.makedirs(os.path.join(DATA_PATH, "pdbs"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_threads(pdb_ids, download_cif, threads)


# STEP 4: process all cif files
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def process_cifs(csv_path, processes):
    """
    Processes all CIF files specified in the CSV file and converts them to into pandas
    dataframes that are saved in binary format.

    This function:
    1. Reads PDB IDs from the provided CSV file
    2. Locates the corresponding CIF files in the data/pdbs directory
    3. Processes each CIF file to extract atom coordinates and metadata
    4. Saves the processed data as parquet files in the data/pdbs_dfs directory

    The processing is done in parallel using the specified number of threads.
    """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "pdbs_dfs")):
        log.info("Creating pdbs_dfs directory: " + os.path.join(DATA_PATH, "pdbs_dfs"))
        os.makedirs(os.path.join(DATA_PATH, "pdbs_dfs"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    cif_files = [os.path.join(DATA_PATH, "pdbs", f"{pdb_id}.cif") for pdb_id in pdb_ids]
    run_w_processes_w_batches(cif_files, process_cif, processes)


# STEP 5: generate dssr outputs
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def generate_dssr_outputs(csv_path, processes):
    """ """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "dssr_output")):
        log.info(
            "Creating dssr_output directory: " + os.path.join(DATA_PATH, "dssr_output")
        )
        os.makedirs(os.path.join(DATA_PATH, "dssr_output"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_processes_w_batches(pdb_ids, generate_dssr_output, processes)


# STEP 6: process all residues
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def process_residues(csv_path, processes):
    """ """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "jsons", "residues")):
        log.info(
            "Creating residues directory: "
            + os.path.join(DATA_PATH, "jsons", "residues")
        )
        os.makedirs(os.path.join(DATA_PATH, "jsons", "residues"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_processes_w_batches(pdb_ids, process_residues_in_pdb, processes)


# STEP 7: process all chains
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def process_chains(csv_path, processes):
    """ """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "jsons", "chains")):
        log.info(
            "Creating chains directory: " + os.path.join(DATA_PATH, "jsons", "chains")
        )
        os.makedirs(os.path.join(DATA_PATH, "jsons", "chains"))
    if not os.path.exists(os.path.join(DATA_PATH, "jsons", "protein_chains")):
        log.info(
            "Creating protein chains directory: "
            + os.path.join(DATA_PATH, "jsons", "protein_chains")
        )
        os.makedirs(os.path.join(DATA_PATH, "jsons", "protein_chains"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_processes_w_batches(pdb_ids, process_chains_in_pdb, processes)


# STEP 8: handle ligand identification
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing ligand info.")
def ligand_identification(csv_path, processes, overwrite):
    """ """
    setup_logging()
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    generate_ligand_info(pdb_ids, processes, overwrite)


# STEP XXX: find cWW basepairs
# generate valid cww pairs


# STEP 8: process all interactions
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def process_interactions(csv_path, processes):
    """ """
    setup_logging()
    # Ensure all required directories exist
    required_dirs = [
        os.path.join(DATA_PATH, "jsons", "hbonds"),
        os.path.join(DATA_PATH, "jsons", "basepairs"),
        os.path.join(DATA_PATH, "dataframes", "hbonds"),
        os.path.join(DATA_PATH, "dataframes", "basepairs"),
    ]

    for directory in required_dirs:
        if not os.path.exists(directory):
            log.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            log.debug(f"Directory already exists: {directory}")
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_processes_w_batches(pdb_ids, process_interactions_in_pdb, processes)


# STEP 9: generate motifs
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def generate_motifs(csv_path, processes):
    """ """
    setup_logging()
    if not os.path.exists(os.path.join(DATA_PATH, "jsons", "motifs")):
        log.info(
            "Creating motifs directory: " + os.path.join(DATA_PATH, "jsons", "motifs")
        )
        os.makedirs(os.path.join(DATA_PATH, "jsons", "motifs"))
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    run_w_processes_w_batches(pdb_ids, generate_motifs_in_pdb, processes)


# OPTIONAL STEP: get pdb info
# get pdb titles, num of residues, num proteins, num ligands, num hbonds, num basepairs, num motifs
# etc, contains non-redundant set, etc
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def get_pdb_info(csv_path):
    """ """
    setup_logging()
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    df_titles = pd.DataFrame(get_pdb_titles_batch(pdb_ids, 15))
    df = df.merge(df_titles, on="pdb_id")
    df.to_json("pdb_titles.json", orient="records")


if __name__ == "__main__":
    cli()
