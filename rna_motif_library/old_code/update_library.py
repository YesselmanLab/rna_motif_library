import csv
import json
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
from rna_motif_library.tert_contacts import (
    import_tert_contact_csv,
    import_residues_csv,
    update_unknown_motifs,
    find_unique_tert_contacts,
    print_tert_contacts_to_cif,
)
from rna_motif_library.logger import get_logger
from rna_motif_library.motif import get_motifs

log = get_logger("update_library")


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
    df = pd.read_csv(
        csv_path, header=None, names=["equivalence_class", "represent", "class_members"]
    )

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
        pdb_name = row.represent.split("|")[0]
        out_path = os.path.join(pdb_dir, f"{pdb_name}.cif")

        if os.path.isfile(out_path):
            return  # Skip this row because the file is already downloaded
        try:
            wget.download(
                f"https://files.rcsb.org/download/{pdb_name}.cif", out=out_path
            )
        except Exception as e:
            tqdm.write(f"Failed to download {pdb_name}: {e}")

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
    validate_and_regenerate_invalid_json_files(out_path, pdb_dir)


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


def generate_motif_files(limit=None, pdb_name=None, directory=None) -> None:
    """
    Processes PDB files to extract and analyze motif interactions, storing detailed outputs.

    Args:
        limit (int): number of PDBs to process
        pdb_name (str): which specific PDB to process (both are entered via command line)

    Returns:
        None

    """
    if directory is not None:
        pdb_dir = directory
    else:
        pdb_dir = os.path.join(LIB_PATH, "data/pdbs/")
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))
    log.info(f"there are {len(pdbs)} files in directory {pdb_dir}")

    if pdb_name is not None:
        log.info(f"Processing PDB: {pdb_name}")
        pdb_name_path = os.path.join(pdb_dir, str(pdb_name) + ".cif")
        pdbs = [pdb_name_path]
        if not os.path.exists(pdb_name_path):
            log.error(f"The provided PDB '{pdb_name}' doesn't exist.")
            log.error("Make sure to run DSSR and SNAP first before generating motifs.")
            log.error("Exiting run.")
            exit(1)

    # Define directories for output
    motif_dir = os.path.join(DATA_PATH, "motifs")
    csv_dir = os.path.join(DATA_PATH, "out_csvs")
    out_json_dir = os.path.join(DATA_PATH, "out_json")
    os.makedirs(motif_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)

    count = 0
    motifs_per_pdb = []
    for pdb_path in pdbs:
        print(pdb_path)
        count += 1
        # Keep processed files under the limit if specified
        if limit is not None and count > limit:
            break
        # Limit processing to specified PDB
        name = os.path.basename(pdb_path)[:-4]
        if (pdb_name != None) and (name != pdb_name):
            break

        built_motifs = get_motifs(name)
        # we can keep this as is it's not too big a CSV I think
        motifs_per_pdb.append(built_motifs)
    exit()
    # dssr_hbonds.print_residues_in_motif_to_csv(motifs_per_pdb, csv_dir)

    # Dump motifs to JSON
    for motif_list in motifs_per_pdb:
        motif_dicts = []
        for motif in motif_list:
            # First take the strands; keep auth_atom_id, cartn_xyz, group_PDB, id
            strands = motif.strands
            for strand in strands:
                # print(type(strand)) a list
                # print(strand) # list of residues
                for residue in strand:
                    pdb_df = residue.pdb
                    new_pdb = pdb_df[
                        [
                            "group_PDB",
                            "id",
                            "auth_atom_id",
                            "Cartn_x",
                            "Cartn_y",
                            "Cartn_z",
                        ]
                    ]
                    residue.pdb = new_pdb
            # Once all the strands are updated, load into JSON
            motif_dict = motif.to_dict()
            motif_dicts.append(motif_dict)
        json_out_path = os.path.join(
            DATA_PATH, "out_json", f"{str(motif.pdb)}_motifs.json"
        )
        with open(json_out_path, "w") as file:
            json.dump(motif_dicts, file)  # Use indent for pretty-printing

    motif_interaction_data_by_type_to_csv(csv_dir)


def motif_interaction_data_by_type_to_csv(csv_dir: str) -> None:
    """
    Sends motif/interaction data (by residue type) to CSV.

    Args:
        csv_dir (str): Directory where CSVS are output.

    Returns:
        None

    """
    # Interaction types, load into CSV as header
    hbond_vals = [
        "base:base",
        "base:sugar",
        "base:phos",
        "sugar:base",
        "sugar:sugar",
        "sugar:phos",
        "phos:base",
        "phos:sugar",
        "phos:phos",
        "base:aa",
        "sugar:aa",
        "phos:aa",
    ]

    # Assuming 'csv_dir' is defined and the path to the CSV files is set
    interactions_df = pd.read_csv(os.path.join(csv_dir, "single_motif_interaction.csv"))

    # Initialize a list to hold the rows for the final DataFrame
    rows = []

    # Group the DataFrame by 'motif_name'
    grouped = interactions_df.groupby("motif_name")

    # Iterate over each group
    for motif_name, group in grouped:
        # Initialize a dictionary to hold the counts for the current motif_name
        counts_dict = {"motif_name": motif_name}

        # Extract the 'type' from 'motif_name' by splitting the string by "."
        counts_dict["type"] = motif_name.split(".")[0]

        # Iterate over each hbond_val to count occurrences
        for hbond in hbond_vals:
            col1, col2 = hbond.split(":")
            # Count the occurrences of this specific combination within the group
            count = group[(group["type_1"] == col1) & (group["type_2"] == col2)].shape[
                0
            ]
            counts_dict[hbond] = count

        # Append the dictionary to the rows list
        rows.append(counts_dict)

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file, including the new 'type' column
    result_df.to_csv(os.path.join(csv_dir, "interactions.csv"), index=False)


def find_tertiary_contacts() -> None:
    """
    Finds and processes tertiary contacts from the found potential tertiary contacts.

    Returns:
        None

    """
    csv_dir = os.path.join(LIB_PATH, "data", "out_csvs")
    # Get motif residue dictionary
    motif_residue_dict = import_residues_csv(csv_dir)
    # Extract potential tert contact DF
    potential_tert_contact_df = import_tert_contact_csv(csv_dir)
    # And find motifs involved in tertiary contacts
    tert_contact_df = update_unknown_motifs(
        potential_tert_contact_df, motif_residue_dict
    )
    tert_contact_df.to_csv(
        os.path.join(csv_dir, "all_tert_contact_hbonds.csv"), index=False
    )
    # Now we need to find unique tertiary contacts
    unique_tert_contact_df = find_unique_tert_contacts(tert_contact_df)
    unique_tert_contact_df.to_csv(
        os.path.join(csv_dir, "unique_tert_contacts.csv"), index=False
    )
    print_tert_contacts_to_cif(unique_tert_contact_df=unique_tert_contact_df)


def count_cif_files(directory: str) -> int:
    """
    Recursively count .cif files in the given directory.

    Args:
        directory (str): A string, the path to the directory to count .cif files in.

    Returns:
        cif_count (int): An integer count of .cif files.

    """
    cif_count = 0
    for root, dirs, files in os.walk(directory):
        cif_count += sum(1 for file in files if file.endswith(".cif"))
    return cif_count


def write_counts_to_csv(motif_directory: str, output_csv: str) -> None:
    """
    Writes the counts of .cif files for each subdirectory to a CSV file.

    Args:
        motif_directory (str): A string, the directory containing motif subdirectories.
        output_csv (str): A string, the path to the output CSV file.
    """
    # List subdirectories in the motif directory
    subdirectories = [
        os.path.join(motif_directory, d)
        for d in os.listdir(motif_directory)
        if os.path.isdir(os.path.join(motif_directory, d))
    ]

    # Prepare data to write
    data_to_write = [["motif_type", "count"]]
    for subdirectory in subdirectories:
        cif_count = count_cif_files(subdirectory)
        data_to_write.append([os.path.basename(subdirectory), cif_count])

    # Write data to CSV
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data_to_write)


def validate_and_regenerate_invalid_json_files(out_path: str, pdb_dir: str):
    """
    Validates all JSON files in the output directory. If a JSON file is invalid,
    it is deleted and regenerated from the corresponding PDB file.

    Args:
        out_path (str): Path to the directory containing the JSON output files.
        pdb_dir (str): Path to the directory containing the original PDB files.

    Returns:
        None
    """
    json_files = glob.glob(os.path.join(out_path, "*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as file:
                json.load(file)  # Try to load the JSON file
        except (json.JSONDecodeError, IOError):
            print(f"Invalid JSON detected: {json_file}. Regenerating...")
            os.remove(json_file)  # Delete the invalid JSON file

            # Regenerate the JSON file from the corresponding PDB file
            pdb_name = os.path.basename(json_file)[:-5] + ".cif"
            pdb_path = os.path.join(pdb_dir, pdb_name)
            json_out_path = os.path.join(out_path, os.path.basename(json_file))

            # Writes raw JSON data
            write_dssr_json_output_to_file(DSSR_EXE, pdb_path, json_out_path)
            print(f"Regenerated: {json_out_path}")
