import csv
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor

import wget
import glob
import os
import threading
import concurrent.futures

import pandas as pd
from tqdm import tqdm

from pydssr.dssr import write_dssr_json_output_to_file

from rna_motif_library.snap import generate_out_file
from rna_motif_library.settings import LIB_PATH, DSSR_EXE, DATA_PATH
from rna_motif_library.util import get_cached_path
from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_motifs, save_motifs_to_json


log = get_logger("update-library")


def download_cif_files(csv_path: str, threads: int) -> None:
    """
    Downloads CIF files based on a CSV that specifies the non-redundant set.

    Args:
        csv_path (str): The path to the CSV file that contains data about which PDB files to download.
        threads (int): number of threads to use

    Returns:
        None

    """
    pdb_dir = DATA_PATH + "/pdbs/"
    os.makedirs(pdb_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    def download_pdbs(row):
        """
        Downloads a PDB file based on the given row.

        Args:
            row (pandas.Series): The row containing information about the PDB file.

        Returns:
            None: If the file is already downloaded, the function returns None.

        Raises:
            Exception: If there is an error while downloading the PDB file.

        """
        pdb_id = row.pdb_id
        out_path = os.path.join(pdb_dir, f"{pdb_id}.cif")

        if os.path.isfile(out_path):
            return  # Skip this row because the file is already downloaded
        try:
            wget.download(f"https://files.rcsb.org/download/{pdb_id}.cif", out=out_path)
        except Exception as e:
            tqdm.write(f"Failed to download {pdb_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        list(
            tqdm(executor.map(download_pdbs, df.itertuples(index=False)), total=len(df))
        )

    # Clean up files with parentheses in their names (duplicates)
    files_with_parentheses = glob.glob(os.path.join(pdb_dir, "*(*.cif"))
    for file in files_with_parentheses:
        os.remove(file)


def get_dssr_files(threads: int, directory: str) -> None:
    """
    Runs DSSR on PDB files to extract and store secondary structure information in JSON format.

    Args:
        threads (int): number of threads to run on
        directory (str): directory of PDBs to process

    Returns:
        None

    """
    if directory is not None:
        pdb_dir = directory
    else:
        pdb_dir = LIB_PATH + "/data/pdbs/"
    out_path = LIB_PATH + "/data/dssr_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))
    count = 0
    lock = threading.Lock()

    def process_pdb(pdb_path):
        nonlocal count
        name = os.path.basename(pdb_path)[:-4]
        json_out_path = os.path.join(out_path, name + ".json")

        if os.path.isfile(json_out_path):
            return 0  # File already processed, no need to increment count
        write_dssr_json_output_to_file(DSSR_EXE, pdb_path, json_out_path)

        with lock:
            count += 1
            log.info(f"Processed {count} PDBs {name}")

        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(process_pdb, pdbs)
    # TODO this should be put into the process_pdb function above to speed up with threading
    # validate_and_regenerate_invalid_json_files(out_path, pdb_dir)


def get_snap_files(threads: int, directory: str) -> None:
    """
    Runs snap to extract RNP interactions for each PDB file and stores the results in .out files.

    Args:
        threads (int): number of threads to run on

    Returns:
        None

    """
    if directory is not None:
        pdb_dir = directory
    else:
        pdb_dir = LIB_PATH + "/data/pdbs/"
    out_path = LIB_PATH + "/data/snap_output/"

    # Ensure the output directory exists
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok=True)

    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))

    def process_pdb(pdb_path):
        name = os.path.basename(pdb_path)[:-4]
        out_file = os.path.join(out_path, f"{name}.out")

        if os.path.isfile(out_file):
            return f"{name}.out ALREADY EXISTS"

        log.info(f"Processing {pdb_path}")
        generate_out_file(pdb_path, out_file)
        return f"{name}.out GENERATED"

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(process_pdb, pdbs))

    # Count the results
    already_exists_count = sum(1 for result in results if "ALREADY EXISTS" in result)
    generated_count = sum(1 for result in results if "GENERATED" in result)

    log.info(f"{already_exists_count} files already existed.")
    log.info(f"{generated_count} new .out files generated.")


def generate_motif_files(pdb_ids: List[str]) -> None:
    """
    Processes PDB files to extract and analyze motif interactions, storing detailed outputs.
    """
    motif_dir = os.path.join(DATA_PATH, "jsons", "motifs")
    os.makedirs(motif_dir, exist_ok=True)

    for pdb_id in pdb_ids:
        motifs = get_motifs(pdb_id)
        save_motifs_to_json(motifs, get_cached_path(pdb_id, "motifs"))
