import time
import warnings
import os

import click

from rna_motif_library import settings
from rna_motif_library.logger import setup_logging, get_logger
import update_library
from update_library import download_cif_files

log = get_logger("cli")

@click.group()
def cli():
    pass


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(threads):
    """
    Downloads CIFs specified in the CSV from the RCSB PDB database.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
    setup_logging()
    warnings.filterwarnings("ignore")
    start_time = time.time()
    csv_directory = os.path.join(settings.LIB_PATH, "data/csvs/")
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    csv_path = os.path.join(csv_directory, csv_files[0])
    update_library.download_cif_files(csv_path, threads)
    download_cif_files(csv_path, threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    log.info(
        "Download started at " + 
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    log.info(
        "Download finished at " +
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
    )
    log.info(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
def process_dssr(threads):
    """
    Processes every downloaded PDB with DSSR, extracting the secondary structure into a JSON.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__get_dssr_files(threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(
        "DSSR processing started at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    print(
        "DSSR processing finished at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
    )
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
def process_snap(threads):
    """
    Processes every downloaded PDB with SNAP, extracting RNA-protein interaction data.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__get_snap_files(threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(
        "SNAP processing started at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    print(
        "SNAP processing finished at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
    )
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


@cli.command()  # Set command name
@click.option(
    "--limit", default=None, type=int, help="Limit the number of PDB files processed."
)
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set, without extensions",
)
def generate_motifs(limit, pdb):
    """
    Extracts motifs from source PDB using data from DSSR, and interactions using data from DSSR and SNAP.

    Args:
        limit (int): Number of PDBs to process (defaults to all).
        pdb (str): Specific PDB ID to process (all by default).

    Returns:
        None
    """

    warnings.filterwarnings("ignore")
    start_time = time.time()

    update_library.__generate_motif_files(limit, pdb)

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print(
        "Motif generation started at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    print(
        "Motif generation finished at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
    )
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


@cli.command()  # Set command name
def find_tertiary_contacts():
    """
    Finds tertiary contacts using hydrogen bonding data.

    Args:
        None

    Returns:
        None
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__find_tertiary_contacts()
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print(
        "Tertiary contact discovery started at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    print(
        "Tertiary contact discovery finished at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
    )
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    cli()
