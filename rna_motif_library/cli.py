import time
import warnings
import os

import click

from rna_motif_library import settings
import update_library


@click.group()
def cli():
    pass


@cli.command(name='download_cifs')
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(threads):
    warnings.filterwarnings("ignore")
    start_time = time.time()
    csv_directory = os.path.join(settings.LIB_PATH, "data/csvs/")
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    csv_path = os.path.join(csv_directory, csv_files[0])
    update_library.__download_cif_files(csv_path, threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print("Download started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Download finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")

@cli.command(name='process_dssr')
@click.option("--threads", default=1, help="Number of threads to use.")
def process_dssr(threads):
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__get_dssr_files(threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print("DSSR processing started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("DSSR processing finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")

@cli.command(name='process_snap')
@click.option("--threads", default=1, help="Number of threads to use.")
def process_snap(threads):
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__get_snap_files(threads)
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print("SNAP processing started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("SNAP processing finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")



@cli.command(name='generate_motifs')  # Set command name
@click.option("--limit", default=None, type=int, help="Limit the number of PDB files processed.")
@click.option("--pdb", default=None, type=str, help="Process a specific PDB within the set, without extensions")
def generate_motifs(limit, pdb):
    warnings.filterwarnings("ignore")
    start_time = time.time()

    update_library.__generate_motif_files(limit, pdb)

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print("Motif generation started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Motif generation finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")



@cli.command(name='find_tertiary_contacts')  # Set command name
def find_tertiary_contacts():
    warnings.filterwarnings("ignore")
    start_time = time.time()
    update_library.__find_tertiary_contacts()
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print("Tertiary contact discovery started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Tertiary contact discovery finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")



# I'm probably going to junk this once I get the notebooks running
"""@cli.command(name='make_figures')  # Set command name
def make_figures():
    warnings.filterwarnings("ignore")
    update_library.__final_statistics()"""


if __name__ == "__main__":
    cli()
