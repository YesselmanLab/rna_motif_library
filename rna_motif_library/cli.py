import warnings
import datetime
import os

import click

from rna_motif_library import settings
import update_library


@click.group()
def cli():
    pass


@cli.command(name='download_cifs')  # Set command name
@click.option("--threads", default=1, help="Number of threads to use.")
def download_cifs(threads):
    warnings.filterwarnings("ignore")
    # current_time = datetime.datetime.now()
    # current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    csv_directory = os.path.join(settings.LIB_PATH, "data/csvs/")
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    csv_path = os.path.join(csv_directory, csv_files[0])
    update_library.__download_cif_files(csv_path, threads)


@cli.command(name='process_dssr')  # Set command name
@click.option("--threads", default=1, help="Number of threads to use.")
def process_dssr(threads):
    warnings.filterwarnings("ignore")
    update_library.__get_dssr_files(threads)


@cli.command(name='process_snap')  # Set command name
@click.option("--threads", default=1, help="Number of threads to use.")
def process_snap(threads):
    warnings.filterwarnings("ignore")
    update_library.__get_snap_files(threads)


@cli.command(name='generate_motifs')  # Set command name
@click.option("--threads", default=1, help="Number of threads to use.")
def generate_motifs(threads):
    warnings.filterwarnings("ignore")
    update_library.__generate_motif_files(threads)
    pass


@cli.command(name='find_tertiary_contacts')  # Set command name
@click.option("--threads", default=1, help="Number of threads to use.")
def find_tertiary_contacts(threads):
    warnings.filterwarnings("ignore")
    update_library.__find_tertiary_contacts(threads)



@cli.command(name='make_figures')  # Set command name
def make_figures():
    warnings.filterwarnings("ignore")
    update_library.__final_statistics()


if __name__ == "__main__":
    cli()
