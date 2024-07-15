from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError

import wget
import glob
import os
import datetime
import warnings
import threading
import concurrent.futures

from typing import Dict
import pandas as pd

import settings
import snap
import dssr
import tertiary_contacts
import figure_plotting
from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS

# list of residue types to filter out
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
    """Class to override standard behavior for handling mmCIF files in Pandas,
    particularly to address inconsistencies between ATOM and HETATM records."""

    def _construct_df(self, text: str) -> pd.DataFrame:
        """Constructs a DataFrame from mmCIF text.

        Args:
            text: The mmCIF file content as a string.

        Returns:
            A combined DataFrame of ATOM and HETATM records.
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


def __safe_mkdir(directory: str) -> None:
    """Safely creates a directory if it does not already exist.

    Args:
        directory: The path of the directory to create.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def __file_exists_in_dir(filename, directory):
    """
    Function to check if file exists in directory or subdirectories

    :param filename: name of file to search for
    :param directory: directory to look inside (includes subfolders)
    :return: returns Boolean (does the file exist or not?)
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return True
    return False


def __download_cif_files(csv_path: str, threads: int) -> None:
    """Downloads CIF files based on a CSV that specifies the non-redundant set.

    Args:
        csv_path: The path to the CSV file that contains data about which PDB files to download.
        threads: number of threads to use
    """
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"

    # Ensure the directory exists
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    # Read the CSV
    df = pd.read_csv(csv_path, header=None, names=["equivalence_class", "represent", "class_members"])

    def download_pdbs(row):
        pdb_name = row.represent.split("|")[0]
        out_path = os.path.join(pdb_dir, f"{pdb_name}.cif")

        if os.path.isfile(out_path):
            print(f"{pdb_name} ALREADY DOWNLOADED!")
        else:
            print(f"{pdb_name} DOWNLOADING")
            wget.download(f"https://files.rcsb.org/download/{pdb_name}.cif", out=out_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Map each row to the download function
        executor.map(download_pdbs, df.itertuples(index=False))

    # Clean up files with parentheses in their names (duplicates)
    files_with_parentheses = glob.glob(os.path.join(pdb_dir, "*(*.cif"))
    for file in files_with_parentheses:
        os.remove(file)
        print(f"Removed file: {file}")


'''def __download_cif_files(csv_path: str, threads: int) -> None:
    """Downloads CIF files based on a CSV that specifies the non-redundant set.

    Args:
        csv_path: The path to the CSV file that contains data about which PDB files to download.
        threads: The number of threads to use for downloading.
    """
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"

    # Ensure the directory exists
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    # Define the structure of the CSV file
    column_names = ["equivalence_class", "represent", "class_members"]

    # Read the CSV
    df = pd.read_csv(csv_path, header=None, names=column_names)

    def download_pdb(row):
        pdb_name = row[1].split("|")[0]  # Access 'represent' by index since it's the second column
        out_path = os.path.join(pdb_dir, f"{pdb_name}.cif")

        if os.path.isfile(out_path):
            return f"{pdb_name} ALREADY DOWNLOADED"

        print(f"{pdb_name} DOWNLOADING")
        wget.download(f"https://files.rcsb.org/download/{pdb_name}.cif", out=out_path)
        return f"{pdb_name} DOWNLOADED"

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = executor.map(download_pdb, df.itertuples(index=False))

    already_downloaded = sum(1 for result in results if 'ALREADY DOWNLOADED' in result)
    print(f"{already_downloaded} PDBs already downloaded!")

    # Clean up files with parentheses in their names
    files_with_parentheses = glob.glob(os.path.join(pdb_dir, "*(*.cif"))
    for file in files_with_parentheses:
        os.remove(file)
        print(f"Removed file: {file}")

    # Count remaining .cif files
    remaining_files = glob.glob(os.path.join(pdb_dir, "*.cif"))
    print(f"Total .cif files after cleanup: {len(remaining_files)}")
'''


def __get_dssr_files(threads: int) -> None:
    """Runs DSSR on PDB files to extract and store secondary structure information in JSON format."""
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    dssr_path = settings.DSSR_EXE
    out_path = settings.LIB_PATH + "/data/dssr_output/"

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
        write_dssr_json_output_to_file(dssr_path, pdb_path, json_out_path)

        with lock:
            count += 1
            print(count, pdb_path)

        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(process_pdb, pdbs)

    print(f"{count} PDB files processed")


def __get_snap_files(threads: int) -> None:
    """Runs snap to extract RNP interactions for each PDB file and stores the results in .out files."""
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    out_path = settings.LIB_PATH + "/data/snap_output/"

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
        snap.__generate_out_file(pdb_path, out_file)  # Call to generate the .out file
        return f"{name}.out GENERATED"

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(process_pdb, pdbs))

    # Count the results
    already_exists_count = sum(1 for result in results if 'ALREADY EXISTS' in result)
    generated_count = sum(1 for result in results if 'GENERATED' in result)

    print(f"{already_exists_count} files already existed.")
    print(f"{generated_count} new .out files generated.")


def __generate_motif_files(errored_count: int) -> None:
    """Processes PDB files to extract and analyze motif interactions, storing detailed outputs."""
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))

    # Define directories for output
    motif_dir = os.path.join("motifs", "nways", "all")
    __safe_mkdir(motif_dir)

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
    with open("interactions.csv", "w") as f_inter_overview, open(
            "interactions_detailed.csv", "w"
    ) as f_inter, open("motif_residues_list.csv", "w") as f_residues, open(
        "twoway_motif_list.csv", "w"
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
            if count < errored_count:
                continue

            name = os.path.basename(pdb_path)[:-4]
            print(f"{count}, {pdb_path}, {name}")
            # if name != "7EQG":
            #    continue
            json_path = os.path.join(
                settings.LIB_PATH, "data/dssr_output", f"{name}.json"
            )
            rnp_out_path = os.path.join(
                settings.LIB_PATH, "data/snap_output", f"{name}.out"
            )
            rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)
            rnp_data = [
                (
                    interaction.nt_atom.split("@")[1],
                    interaction.aa_atom.split("@")[1],
                    interaction.nt_atom.split("@")[0],
                    interaction.aa_atom.split("@")[0],
                    str(interaction.dist),
                    interaction.type.split(":")[0],
                    interaction.type.split(":")[1],
                )
                for interaction in rnp_interactions
            ]

            pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
            # In case of bugged JSON file
            while True:
                try:
                    (
                        motifs,
                        motif_hbonds,
                        motif_interactions,
                        hbonds_in_motif,
                    ) = dssr.get_motifs_from_structure(json_path)
                    break
                except JSONDecodeError:
                    os.remove(json_path)
                    __get_dssr_files(1)
                    (
                        motifs,
                        motif_hbonds,
                        motif_interactions,
                        hbonds_in_motif,
                    ) = dssr.get_motifs_from_structure(json_path)

            hbonds_in_motif.extend(rnp_data)
            unique_inter_motifs = list(set(hbonds_in_motif))

            for m in motifs:
                if m.name.split(".")[0] not in [
                    "TWOWAY",
                    "NWAY",
                    "HAIRPIN",
                    "HELIX",
                    "SSTRAND",
                ]:
                    continue
                interactions = motif_interactions.get(m.name, None)
                dssr.write_res_coords_to_pdb(
                    m.nts_long,
                    interactions,
                    pdb_model,
                    os.path.join(motif_dir, m.name),
                    unique_inter_motifs,
                    f_inter,
                    f_residues,
                    f_twoways,
                    f_inter_overview,
                )


def __find_tertiary_contacts():
    """
        Finds and processes tertiary contacts in RNA motifs.print("Finding tertiary contacts...")
    # loads into a dictionary all residues in given motifs; motif names as keys
        This function loads residues from motifs, reads interactions from a CSV file,motif_residues_dict = tertiary_contacts.load_motif_residues(motif_residues_csv_path="motif_residues_list.csv")
        groups the interactions by motif names, finds tertiary contacts, identifies unique# Read the interactions CSV file into a DataFrame and obtain its contents
        tertiary contacts, removes duplicates, prints the results to a CSV file, and plots histograms.interactions_csv_df = pd.read_csv("interactions_detailed.csv")
    # Group the DataFrame by the 'name' column # and Convert the DataFrameGroupBy back into individual DataFrames
        Args:grouped_interactions_csv_df = interactions_csv_df.groupby("name")
            None
    # Iterate over each motif (group) to find tertiary contacts
        Returns:tertiary_contacts.find_tertiary_contacts(interactions_from_csv=grouped_interactions_csv_df,
            None                                         list_of_res_in_motifs=motif_residues_dict)
    """
    interactions_from_csv = pd.read_csv("interactions_detailed.csv")
    grouped_interactions_csv_df = interactions_from_csv.groupby("name")
    motif_residues_csv_path = "motif_residues_list.csv"
    motif_residues_dict = tertiary_contacts.load_motif_residues(motif_residues_csv_path)
    tertiary_contacts.find_tertiary_contacts(
        interactions_from_csv=grouped_interactions_csv_df,
        list_of_res_in_motifs=motif_residues_dict,
    )
    tertiary_contacts.find_unique_tertiary_contacts()
    unique_tert_contact_df = tertiary_contacts.delete_duplicate_contacts()
    tertiary_contacts.print_tert_contacts_to_csv(unique_tert_contact_df)
    tertiary_contacts.plot_tert_histograms(unique_tert_contact_df)


# calculate some final statistics
def __final_statistics():
    """
    Calculate and plot final statistics for RNA motifs and interactions.

    This function generates various plots for motif counts, hairpin counts, helix counts,
    single-strand counts, and tertiary contact counts. It also filters and groups hydrogen
    bond interactions and plots the results.

    Args:
        None

    Returns:
        None
    """
    motif_directory = os.path.join(settings.LIB_PATH, "motifs")
    tert_motif_directory = (
        os.path.join(settings.LIB_PATH, "tertiary_contacts")
    )
    tert_contact_csv_directory = "unique_tert_contacts.csv"

    figure_plotting.plot_motif_counts(motif_directory=motif_directory)
    figure_plotting.plot_hairpin_counts(motif_directory=motif_directory)
    figure_plotting.plot_helix_counts(motif_directory=motif_directory)
    figure_plotting.plot_sstrand_counts(motif_directory=motif_directory)
    figure_plotting.plot_tert_contact_counts(tert_motif_directory=tert_motif_directory)
    figure_plotting.plot_tert_contact_type_counts(
        tert_contact_csv_directory=tert_contact_csv_directory
    )

    print("Plotting heatmaps...")
    # Read the CSV data into a DataFrame
    csv_path = "twoway_motif_list.csv"
    figure_plotting.plot_twoway_size_heatmap(csv_path=csv_path)
    hbond_df_unfiltered = pd.read_csv("interactions_detailed.csv")
    # also delete res_1_name and res_2_name where they are hairpins less than 3
    # Create an empty DataFrame to store the filtered data
    filtered_data = []
    # Iterate through each row in the unfiltered DataFrame
    for index, row in hbond_df_unfiltered.iterrows():
        # Split motif_1 and motif_2 by "."
        motif_1_split = row["name"].split(".")

        # Check conditions for deletion
        if motif_1_split[0] == "HAIRPIN" and 0 < float(motif_1_split[2]) < 3:
            continue
        else:
            # Keep the row by appending it to the filtered_data list
            filtered_data.append(row)
    # Create a new DataFrame with the filtered data
    hbond_df = pd.DataFrame(filtered_data)
    # Reset the index of the new DataFrame
    hbond_df.reset_index(drop=True, inplace=True)
    # now delete all non-canonical residues (if we need to keep DA/DC/DU/etc here is where to do it)
    filtered_hbond_df = hbond_df[
        hbond_df["res_1_name"].isin(canon_res_list)
        & hbond_df["res_2_name"].isin(canon_res_list)
        ]

    # reverse orders are sorted into the same group
    filtered_hbond_df["res_atom_pair"] = filtered_hbond_df.apply(
        lambda row: tuple(
            sorted(
                [(row["res_1_name"], row["atom_1"]), (row["res_2_name"], row["atom_2"])]
            )
        ),
        axis=1,
    )

    # next, group by (res_1_name, res_2_name) as well as by atoms involved in the interaction
    # grouped_hbond_df = filtered_hbond_df.groupby(["res_1_name", "res_2_name", "atom_1", "atom_2"])
    grouped_hbond_df = filtered_hbond_df.groupby(["res_atom_pair"])
    figure_plotting.plot_present_hbonds(grouped_hbond_df=grouped_hbond_df)


def main():
    warnings.filterwarnings("ignore")
    current_time = datetime.datetime.now()
    start_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("!!!!! CIF FILES DOWNLOADED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Download finished on", time_string)
    # __get_dssr_files()
    print("!!!!! DSSR PROCESSING FINISHED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("DSSR processing finished on", time_string)
    # __get_snap_files()
    print("!!!!! SNAP PROCESSING FINISHED !!!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("SNAP processing finished on", time_string)
    # __generate_motif_files()
    print("!!!!! MOTIF EXTRACTION FINISHED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Motif extraction finished on", time_string)
    # __find_tertiary_contacts()
    print("!!!!! TERTIARY CONTACT PROCESSING FINISHED !!!!!")
    print("Plotting data...")
    # __final_statistics()
    print("!!!!! PLOTS COMPLETED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Job started on", start_time_string)
    print("Job finished on", time_string)


if __name__ == "__main__":
    main()
