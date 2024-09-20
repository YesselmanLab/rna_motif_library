import time
import warnings
import os
import click
import sys
import functools
import pandas as pd
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.logger import setup_logging, get_logger
from update_library import (
    get_dssr_files,
    get_snap_files,
    download_cif_files,
    find_tertiary_contacts,
    generate_motif_files,
)

log = get_logger("cli")


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
        func_name = func.__name__
        setup_logging()
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
@log_and_setup
def download_cifs(threads):
    """
    Downloads CIFs specified in the CSV from the RCSB PDB database.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
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
@log_and_setup
def process_dssr(threads):
    """
    Processes every downloaded PDB with DSSR, extracting the secondary structure into a JSON.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
    warnings.filterwarnings("ignore")
    get_dssr_files(threads)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@log_and_setup
def process_snap(threads):
    """
    Processes every downloaded PDB with SNAP, extracting RNA-protein interaction data.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
    warnings.filterwarnings("ignore")
    get_snap_files(threads)


@cli.command()
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit the number of PDB files processed (defaults to all)."
)
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all)."
)
@log_and_setup
def generate_motifs(limit, pdb):
    """
    Extracts motifs from source PDB using data from DSSR, and interactions using data from DSSR and SNAP.

    Args:
        limit (int): Number of PDBs to process (defaults to all).
        pdb (str): Specific PDB ID to process (all by default).
        threads (int): number of threads to use (default: 1)

    Returns:
        None

    """
    warnings.filterwarnings("ignore")
    generate_motif_files(limit=limit, pdb_name=pdb)


@cli.command()
@log_and_setup
def load_tertiary_contacts():
    """
    Finds tertiary contacts using hydrogen bonding data.

    Args:
        None

    Returns:
        None

    """
    warnings.filterwarnings("ignore")
    find_tertiary_contacts()


if __name__ == "__main__":
    cli()
