import time
import warnings
import os
import click
import sys
import functools
import json
import glob
import pandas as pd
from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.logger import setup_logging, get_logger
from rna_motif_library.classes import (
    Residue,
    sanitize_x3dna_atom_name,
    X3DNAResidueFactory,
    get_x3dna_res_id,
)
from rna_motif_library.update_library import (
    get_dssr_files,
    get_snap_files,
    download_cif_files,
    generate_motif_files,
)
from rna_motif_library.interactions import get_hbonds_and_basepairs
from rna_motif_library.motif import Motif

# TODO check other types of DSSR classes like kissing loops


log = get_logger("cli")


def get_pdb_codes(pdb: str = None, directory: str = None) -> list:
    """
    Get list of PDB codes based on input parameters.

    Args:
        pdb (str, optional): Single PDB code to process. Defaults to None.
        directory (str, optional): Directory containing PDB files. Defaults to None.

    Returns:
        list: List of PDB codes to process
    """
    pdb_codes = []
    if pdb is not None:
        pdb_codes.append(pdb)
    elif directory is not None:
        pdb_codes = [os.path.basename(file)[:-4] for file in os.listdir(directory)]
    else:
        files = glob.glob(os.path.join(DATA_PATH, "pdbs", "*.cif"))
        for file in files:
            pdb_code = os.path.basename(file)[:-4]
            pdb_codes.append(pdb_code)
    return pdb_codes


def log_and_setup(func):
    """
    Decorator to set up logging and log the start and end of the function execution.

    Args:
        func (callable): The function to wrap.

    Returns:
        callable: The wrapped function with logging and setup.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "debug" not in kwargs:
            kwargs["debug"] = False
        func_name = func.__name__
        setup_logging(debug=kwargs["debug"])
        log.info("Ran at commandline as: %s", " ".join(sys.argv))
        start_time = time.time()
        log.info("Starting time: %s" % pd.Timestamp.now())
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            log.info(
                f"{func_name} execution completed. Total runtime: {formatted_time}"
            )
        return result

    return wrapper


@click.group()
def cli():
    pass


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@log_and_setup
def download_cifs(threads, debug):
    """
    Downloads CIFs specified in the CSV from the RCSB PDB database.

    Args:
        threads (int): Number of threads to run on.
        debug (bool): Enable debugging output.

    Returns:
        None

    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    csv_directory = os.path.join(LIB_PATH, "data/csvs/")
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    # TODO specify which CSV to use
    if len(csv_files) == 0:
        log.error(f"No CSV files found in directory: {csv_directory}")
        return
    csv_path = os.path.join(csv_directory, csv_files[0])
    download_cif_files(csv_path, threads)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--directory", default=None, help="Directory to PDBs used")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@log_and_setup
def process_dssr(threads, directory, debug):
    """
    Processes every downloaded PDB with DSSR, extracting the secondary structure into a JSON.

    Args:
        threads (int): Number of threads to run on.
        directory (str): Directory to use for processing.
        Directory = "/notebooks/distributed_sets/set_i/"

    Returns:
        None

    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    get_dssr_files(threads, directory)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--directory", default=None, help="Directory of PDBs to process")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@log_and_setup
def process_snap(threads, directory, debug):
    """
    Processes every downloaded PDB with SNAP, extracting RNA-protein interaction data.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    get_snap_files(threads, directory)


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@log_and_setup
def process_residues(pdb, directory, debug):
    """
    Processes residues from source PDB using data from DSSR.
    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "residues"), exist_ok=True)
    pdb_codes = get_pdb_codes(pdb, directory)
    log.info(f"Processing {len(pdb_codes)} PDBs")
    for pdb_code in pdb_codes:
        df_atoms = pd.read_parquet(
            os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_code}.parquet")
        )
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
            x3dna_res_id = get_x3dna_res_id(res_name, res_num, chain_id, ins_code)
            x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
            residues[x3dna_res_id] = Residue.from_x3dna_residue(
                x3dna_res, atom_names, coords
            )
        # Save residues to json file
        residues_json_path = os.path.join(
            DATA_PATH, "jsons", "residues", f"{pdb_code}.json"
        )
        with open(residues_json_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in residues.items()}, f)


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.option("--overwrite", is_flag=True, help="Overwrite existing interactions")
@log_and_setup
def process_interactions(pdb, directory, debug, overwrite):
    """
    Processes interactions from source PDB using data from DSSR and interactions using data from SNAP.
    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "jsons", "basepairs"), exist_ok=True)
    pdb_codes = get_pdb_codes(pdb, directory)
    log.info(f"Processing {len(pdb_codes)} PDBs")
    for pdb_code in pdb_codes:
        hbonds, basepairs = get_hbonds_and_basepairs(pdb_code)
        log.info(
            f"Processed {pdb_code} with {len(hbonds)} hbonds and {len(basepairs)} basepairs"
        )
        # Save hbonds to json file
        hbonds_json_path = os.path.join(
            DATA_PATH, "jsons", "hbonds", f"{pdb_code}.json"
        )
        # os.makedirs(os.path.dirname(hbonds_json_path), exist_ok=True)
        # with open(hbonds_json_path, "w") as f:
        #    json.dump([hbond.to_dict() for hbond in hbonds], f)

        # Save basepairs to json file
        basepairs_json_path = os.path.join(
            DATA_PATH, "jsons", "basepairs", f"{pdb_code}.json"
        )
        os.makedirs(os.path.dirname(basepairs_json_path), exist_ok=True)
        with open(basepairs_json_path, "w") as f:
            json.dump([bp.to_dict() for bp in basepairs], f)


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@log_and_setup
def generate_motifs(pdb, directory, debug):
    """
    Extracts motifs from source PDB using data from DSSR, and interactions using data from DSSR and SNAP.

    Args:
        limit (int): Number of PDBs to process (defaults to all).
        pdb (str): Specific PDB ID to process (all by default).
        threads (int): number of threads to use (default: 1)

    Returns:
        None

    """
    if debug:
        log.info("Debug mode is enabled.")
    warnings.filterwarnings("ignore")
    pdb_codes = get_pdb_codes(pdb, directory)
    generate_motif_files(pdb_codes)


if __name__ == "__main__":
    cli()
