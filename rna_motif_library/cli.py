import warnings
import datetime
import os

import click

from rna_motif_library import settings
from rna_motif_library.update_library import download_cif_files


@click.group
def cli():
    pass


@cli.command
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(threads):
    warnings.filterwarnings("ignore")
    current_time = datetime.datetime.now()
    start_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    csv_directory = os.path.join(settings.LIB_PATH, "data/csvs/")
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    csv_path = os.path.join(csv_directory, csv_files[0])
    download_cif_files(csv_path, threads)


if __name__ == "__main__":
    cli()
