import csv
from concurrent.futures import ThreadPoolExecutor

import wget
import glob
import os
import threading
import concurrent.futures

from typing import Dict
import pandas as pd
from tqdm import tqdm

from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS

from rna_motif_library import tert_contacts
from rna_motif_library.snap import generate_out_file
from rna_motif_library import dssr
from rna_motif_library import figure_plotting
from rna_motif_library.settings import LIB_PATH, DSSR_EXE
from rna_motif_library.figure_plotting import safe_mkdir

canon_res_list = [
    "A",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "C",
    "G",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "U",
    "VAL",
]


class PandasMmcifOverride(PandasMmcif):
    """
    Class to override standard behavior for handling mmCIF files in Pandas,
    particularly to address inconsistencies between ATOM and HETATM records.

    """

    def _construct_df(self, text: str) -> pd.DataFrame:
        """
        Constructs a DataFrame from mmCIF text.

        Args:
            text (str): The mmCIF file content as a string.

        Returns:
            combined_df (pd.DataFrame): A combined DataFrame of ATOM and HETATM records.

        """
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data

        df: Dict[str, pd.DataFrame] = {}
        full_df = pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        # Combine ATOM and HETATM records into a single DataFrame
        combined_df = pd.DataFrame(
            full_df[(full_df.group_PDB == "ATOM") | (full_df.group_PDB == "HETATM")]
        )

        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)

        return combined_df  # Return the combined DataFrame


def download_cif_files(csv_path: str, threads: int) -> None:
    """
    Downloads CIF files based on a CSV that specifies the non-redundant set.

    Args:
        csv_path (str): The path to the CSV file that contains data about which PDB files to download.
        threads (int): number of threads to use

    Returns:
        None

    """
    pdb_dir = LIB_PATH + "/data/pdbs/"
    # Ensure the directory exists
    os.makedirs(pdb_dir, exist_ok=True)
    # Read the CSV
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


def get_dssr_files(threads: int) -> None:
    """
    Runs DSSR on PDB files to extract and store secondary structure information in JSON format.

    Args:
        threads (int): number of threads to run on

    Returns:
        None

    """
    pdb_dir = LIB_PATH + "/data/pdbs/"
    out_path = LIB_PATH + "/data/dssr_output/"

    # Ensure output directory exists
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

        # Writes raw JSON data
        write_dssr_json_output_to_file(DSSR_EXE, pdb_path, json_out_path)

        with lock:
            count += 1
            print(count, pdb_path)

        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(process_pdb, pdbs)


def get_snap_files(threads: int) -> None:
    """
    Runs snap to extract RNP interactions for each PDB file and stores the results in .out files.

    Args:
        threads (int): number of threads to run on

    Returns:
        None

    """
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

        print(f"Processing {pdb_path}")  # Debug: prints the PDB path being processed
        generate_out_file(pdb_path, out_file)  # Call to generate the .out file
        return f"{name}.out GENERATED"

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(process_pdb, pdbs))

    # Count the results
    already_exists_count = sum(1 for result in results if "ALREADY EXISTS" in result)
    generated_count = sum(1 for result in results if "GENERATED" in result)

    print(f"{already_exists_count} files already existed.")
    print(f"{generated_count} new .out files generated.")


def generate_motif_files(limit=None, pdb_name=None) -> None:
    """
    Processes PDB files to extract and analyze motif interactions, storing detailed outputs.

    Args:
        limit (int): number of PDBs to process
        pdb_name (str): which specific PDB to process (both are entered via command line)

    Returns:
        None

    """
    pdb_dir = os.path.join(LIB_PATH, "data/pdbs/")
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))

    if pdb_name is not None:
        pdb_name_path = os.path.join(pdb_dir, str(pdb_name) + ".cif")
        if not os.path.exists(pdb_name_path):
            print(f"The provided PDB '{pdb_name}' doesn't exist.")
            print("Make sure to run DSSR and SNAP first before generating motifs.")
            print("Exiting run.")
            exit(1)

    # Define directories for output
    motif_dir = os.path.join(LIB_PATH, "data", "motifs")
    csv_dir = os.path.join(LIB_PATH, "data", "out_csvs")
    safe_mkdir(motif_dir)
    safe_mkdir(csv_dir)
    os.makedirs(motif_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Interaction types
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

    # Open files for output
    with open(os.path.join(csv_dir, "interactions.csv"), "w") as f_inter_overview, open(
            os.path.join(csv_dir, "interactions_detailed.csv"), "w"
    ) as f_inter, open(
        os.path.join(csv_dir, "motif_residues_list.csv"), "w"
    ) as f_residues, open(
        os.path.join(csv_dir, "twoway_motif_list.csv"), "w"
    ) as f_twoways:

        # Write headers
        f_inter_overview.write("name,type,size," + ",".join(hbond_vals) + "\n")
        f_inter.write(
            "name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2\n"
        )
        f_residues.write("motif_name,residues\n")
        f_twoways.write("motif_name,motif_type,bridging_nts_0,bridging_nts_1\n")

        count = 0
        for pdb_path in pdbs:
            count += 1
            dssr.process_pdbs(
                count,
                pdb_path,
                limit,
                pdb_name,
                motif_dir,
                f_inter,
                f_residues,
                f_twoways,
                f_inter_overview,
            )

    # When all is said and done need to count number of motifs and print to CSV
    motif_directory = os.path.join("data/motifs")
    safe_mkdir(motif_directory)
    os.makedirs(motif_directory, exist_ok=True)
    output_csv = os.path.join("data/out_csvs/motif_cif_counts.csv")
    write_counts_to_csv(motif_directory, output_csv)

    # Need to print data for every H-bond group
    hbond_df_unfiltered = pd.read_csv("data/out_csvs/interactions_detailed.csv")
    filtered_data = []
    # Iterate through each row in the unfiltered DataFrame
    for index, row in hbond_df_unfiltered.iterrows():
        motif_1_split = row["name"].split(".")
        # Check conditions for deletion
        if motif_1_split[0] == "HAIRPIN" and 0 < float(motif_1_split[2]) < 3:
            continue
        else:
            filtered_data.append(row)
    hbond_df = pd.DataFrame(filtered_data)
    hbond_df.reset_index(drop=True, inplace=True)
    filtered_hbond_df = hbond_df[
        hbond_df["res_1_name"].isin(canon_res_list)
        & hbond_df["res_2_name"].isin(canon_res_list)
        ]
    filtered_hbond_df["res_atom_pair"] = filtered_hbond_df.apply(
        lambda row: tuple(sorted([(row["res_1_name"], row["atom_1"]), (row["res_2_name"], row["atom_2"])])),
        axis=1,
    )
    grouped_hbond_df = filtered_hbond_df.groupby(["res_atom_pair"])
    figure_plotting.save_present_hbonds(grouped_hbond_df=grouped_hbond_df)


def find_tertiary_contacts() -> None:
    """
    Finds and processes tertiary contacts from the resultant motif/interaction data

    Returns:
        None

    """
    csv_dir = os.path.join(LIB_PATH, "data", "out_csvs")

    interactions_from_csv = pd.read_csv(
        os.path.join(csv_dir, "interactions_detailed.csv")
    )
    grouped_interactions_csv_df = interactions_from_csv.groupby("name")
    motif_residues_csv_path = os.path.join(csv_dir, "motif_residues_list.csv")
    motif_residues_dict = tert_contacts.load_motif_residues(motif_residues_csv_path)
    tert_contacts.find_tertiary_contacts(
        interactions_from_csv=grouped_interactions_csv_df,
        list_of_res_in_motifs=motif_residues_dict,
        csv_dir=csv_dir,
    )
    tert_contacts.find_unique_tertiary_contacts(csv_dir=csv_dir)
    unique_tert_contact_df = tert_contacts.delete_duplicate_contacts(csv_dir=csv_dir)
    tert_contacts.print_tert_contacts_to_cif(
        unique_tert_contact_df=unique_tert_contact_df
    )


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
