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
from update_library import (
    get_dssr_files,
    get_snap_files,
    download_cif_files,
    find_tertiary_contacts,
    generate_motif_files,
)
from rna_motif_library.interactions import get_hbonds_and_basepairs

# TODO look at this stuff for the next week or so
# we want the angle not the dihedral angle
# so save the dihedral but need to do angle calcs and rename the dihedral stuff and push it aside
# also, take another look at threading because it shouldn't be so damn slow
# figure out how to use github release
# have a docker image to run on a virtual machine
# we don't have the rights to DSSR or SNAP so we need to add a way to install these things

####
# check for pseudoknot:
# does a strand of a helix exist within another motif (hairpin/sstrand)? (easy check to write)

#### TODO look at this stoff - 10/25/2024 and week or so after and build it out
# if 2 or more strands are within motif A that exist in motif B, then motif A should be rejected

# write a function to generate PDB from residue/motif


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
@click.option("--directory", default=None, help="Directory to PDBs used")
@log_and_setup
def process_dssr(threads, directory):
    """
    Processes every downloaded PDB with DSSR, extracting the secondary structure into a JSON.

    Args:
        threads (int): Number of threads to run on.
        directory (str): Directory to use for processing.
        Directory = "/notebooks/distributed_sets/set_i/"

    Returns:
        None

    """
    warnings.filterwarnings("ignore")
    get_dssr_files(threads, directory)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--directory", default=None, help="Directory of PDBs to process")
@log_and_setup
def process_snap(threads, directory):
    """
    Processes every downloaded PDB with SNAP, extracting RNA-protein interaction data.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
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
def process_interactions(pdb, directory, debug):
    """
    Processes interactions from source PDB using data from DSSR and interactions using data from SNAP.
    """
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "jsons", "basepairs"), exist_ok=True)
    pdb_codes = []
    if pdb is not None:
        pdb_codes.append(pdb)
    elif directory is not None:
        pdb_codes = [os.path.basename(file)[:-4] for file in os.listdir(directory)]
    else:
        pdb_codes = [
            os.path.basename(file)[:-4]
            for file in glob.glob(os.path.join(DATA_PATH, "pdbs", "*.cif"))
        ]
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
        os.makedirs(os.path.dirname(hbonds_json_path), exist_ok=True)
        with open(hbonds_json_path, "w") as f:
            json.dump([hbond.to_dict() for hbond in hbonds], f)

        # Save basepairs to json file
        basepairs_json_path = os.path.join(
            DATA_PATH, "jsons", "basepairs", f"{pdb_code}.json"
        )
        os.makedirs(os.path.dirname(basepairs_json_path), exist_ok=True)
        with open(basepairs_json_path, "w") as f:
            json.dump([bp.to_dict() for bp in basepairs], f)


@cli.command()
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit the number of PDB files processed (defaults to all).",
)
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
def generate_motifs(limit, pdb, directory, debug):
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
    # create a JSON output file for motifs
    json_out_directory = os.path.join(LIB_PATH, "data", "out_json")
    os.makedirs(json_out_directory, exist_ok=True)
    generate_motif_files(limit=limit, pdb_name=pdb, directory=directory)


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


@cli.command()
@log_and_setup
def reload_from_json():
    """
    Reloads motifs from JSON output data stored in 'data/out_json/' directory.
    Processes strands into a new DataFrame structure and generates .cif files.
    """
    directory_path = "data/out_json/"
    output_dir = "data/out_motifs_json"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            full_path = os.path.join(directory_path, filename)
            with open(full_path, "r") as file:
                data = json.load(file)
                # Process each motif entry individually
                for entry in data:
                    df_motif = process_strands(entry["strands"])
                    generate_cif_file(df_motif, entry["motif_name"], output_dir)


def process_strands(strands):
    """
    Converts strands data into a DataFrame with appropriate column names and returns it.
    Handles both single and multiple atom formats within 'pdb' data.
    """
    processed_strands = []
    for strand in strands:
        for residue in strand:
            # Check if 'pdb' data is a single dictionary and convert to list of dictionaries if so
            pdb_data = residue["pdb"]
            if isinstance(pdb_data, dict):  # It's a single atom
                pdb_data = [pdb_data]  # Make it a list of one dictionary

            # Now process pdb_data assuming it's always a list of dictionaries
            pdb_df = pd.DataFrame(pdb_data)
            pdb_df.rename(
                columns={
                    "id": "id",
                    "Cartn_x": "Cartn_x",
                    "Cartn_y": "Cartn_y",
                    "Cartn_z": "Cartn_z",
                },
                inplace=True,
            )
            pdb_df["auth_asym_id"] = residue["chain_id"]
            pdb_df["auth_seq_id"] = residue["res_id"]
            pdb_df["pdbx_PDB_ins_code"] = residue["ins_code"]
            pdb_df["auth_comp_id"] = residue["mol_name"]
            processed_strands.append(pdb_df)

    return pd.concat(processed_strands, ignore_index=True)


def generate_cif_file(df, motif_name, base_dir):
    """
    Generates a .cif file from the DataFrame and saves it into structured directories based on motif properties.
    """

    pre_parts = motif_name.split()
    parts = pre_parts[0].split(".")
    type_of_motif = parts[0]
    size_of_motif = parts[2]
    sequence_of_motif = parts[3]

    # Build directory path
    path = os.path.join(base_dir, type_of_motif, size_of_motif, sequence_of_motif)
    os.makedirs(path, exist_ok=True)

    # File path
    file_path = os.path.join(path, f"{motif_name}.cif")

    # Write CIF
    with open(file_path, "w") as file:
        file.write("data_" + motif_name + "\n")
        file.write(
            "_audit_creation_method 'Generated by DataFrame to CIF conversion'\n"
        )
        file.write("\n")
        file.write("loop_\n")
        file.write("_atom_site.group_PDB\n")
        file.write("_atom_site.id\n")
        file.write("_atom_site.auth_atom_id\n")
        file.write("_atom_site.auth_asym_id\n")
        file.write("_atom_site.auth_seq_id\n")
        file.write("_atom_site.pdbx_PDB_ins_code\n")
        file.write("_atom_site.Cartn_x\n")
        file.write("_atom_site.Cartn_y\n")
        file.write("_atom_site.Cartn_z\n")
        for index, row in df.iterrows():
            file.write(
                "{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}\n".format(
                    row["group_PDB"],
                    row["id"],
                    row["auth_atom_id"],
                    row["auth_asym_id"],
                    row["auth_seq_id"],
                    row["pdbx_PDB_ins_code"],
                    row["Cartn_x"],
                    row["Cartn_y"],
                    row["Cartn_z"],
                )
            )


if __name__ == "__main__":
    cli()
