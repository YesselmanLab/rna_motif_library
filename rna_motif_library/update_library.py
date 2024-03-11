import csv
import glob

import requests
import json
import os
import datetime
import warnings
from typing import Dict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import wget
from collections import Counter

import settings
import snap
import dssr
from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS

# amino acid/canonical residue dictionary
canon_res_dict = {
    'A': 'Adenine',
    'ALA': 'Alanine',
    'ARG': 'Arginine',
    'ASN': 'Asparagine',
    'ASP': 'Aspartic Acid',
    'CYS': 'Cysteine',
    'C': 'Cytosine',
    'G': 'Guanine',
    'GLN': 'Glutamine',
    'GLU': 'Glutamic Acid',
    'GLY': 'Glycine',
    'HIS': 'Histidine',
    'ILE': 'Isoleucine',
    'LEU': 'Leucine',
    'LYS': 'Lysine',
    'MET': 'Methionine',
    'PHE': 'Phenylalanine',
    'PRO': 'Proline',
    'SER': 'Serine',
    'THR': 'Threonine',
    'TRP': 'Tryptophan',
    'TYR': 'Tyrosine',
    'U': 'Uracil',
    'VAL': 'Valine'
}

canon_res_list = ['A', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'C', 'G', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'U', 'VAL']


# Pandas mmCIF override
class PandasMmcifOverride(PandasMmcif):
    def _construct_df(self, text: str):
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data
        df: Dict[str, pd.DataFrame] = {}
        full_df = pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        # Combine ATOM and HETATM records into the same DataFrame, this solves residue deletion
        combined_df = pd.DataFrame(
            full_df[(full_df.group_PDB == "ATOM") | (full_df.group_PDB == "HETATM")])

        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)

        return combined_df  # Return the combined DataFrame


# Safely create a directory if it doesn't exist.
def __safe_mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


# download the redundant set (may not actually work)
def __download_redundant_cif_files():
    # Define the directory to save the PDB files
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    # Define the API endpoints
    search_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={settings.QUERY_TERM}"
    download_url = "https://files.rcsb.org/download/"
    # Perform the search and download the PDB files (actually CIF but screw it)
    response = requests.post(search_url, data=json.dumps(settings.QUERY_TERM))

    #print(response)
    #exit(0)
    results = response.json()["result_set"]
    # iterates over each item in the results list obtained from the search response
    for result in results:
        # extracts the PDB identifier (pdb_id) and constructs a file path to save the CIF
        pdb_id = result["identifier"]
        pdb_file = f"{pdb_dir}/{pdb_id}.cif"
        if os.path.exists(pdb_file):
            print(f"{pdb_id} ALREADY DOWNLOADED")
        else:
            pdb_url = f"{download_url}{pdb_id}.cif"
            print(f"{pdb_id} DOWNLOADING")
            response = requests.get(pdb_url)
            # content of the response is then written to the pdb_file
            with open(pdb_file, "wb") as f:
                f.write(response.content)


def __download_cif_files(csv_path):
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    count = 0

    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    # Specification of column names
    column_names = ["equivalence_class", "represent", "class_members"]

    # Create a temporary directory to store temporary files
    # temp_dir = tempfile.mkdtemp()

    for i, row in pd.read_csv(csv_path, header=None, names=column_names).iterrows():
        spl = row["represent"].split("|")
        pdb_name = spl[0]
        out_path = pdb_dir + f"{pdb_name}.cif"

        path = f"https://files.rcsb.org/download/{pdb_name}.cif"
        if os.path.isfile(out_path):
            count += 1
            # print(pdb_name + " ALREADY DOWNLOADED!")
            continue
        else:
            print(pdb_name + " DOWNLOADING")
        # Download the content to the temporary directory
        wget.download(path, out=out_path)
    print(f"{count} pdbs already downloaded!")


def __get_dssr_files():
    # creates and sets directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    dssr_path = settings.DSSR_EXE
    out_path = settings.LIB_PATH + "/data/dssr_output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 1
    for pdb_path in pdbs:
        s = os.path.getsize(pdb_path)
        print(count, pdb_path, s)  # s = size of file in bytes
        # if s > 10000000:
        #    continue
        name = pdb_path.split("/")[-1][:-4]
        if os.path.isfile(out_path):
            count += 1
            continue
        # writes raw JSON data
        write_dssr_json_output_to_file(
            dssr_path, pdb_path, out_path + "/" + name + ".json"
        )


def __get_snap_files():
    # creates and sets directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    out_path = settings.LIB_PATH + "/data/snap_output"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # scans every CIF file and stores the output in a .out file
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 0
    for pdb_path in pdbs:
        s = os.path.getsize(pdb_path)
        # if s > 10000000:
        #    continue
        print(count, pdb_path)
        name = pdb_path.split("/")[-1][:-4]
        out_file = out_path + "/" + name + ".out"
        if os.path.isfile(out_file):
            count += 1
            continue
        print(pdb_path)
        snap.__generate_out_file(pdb_path, out_file)
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"


def __generate_motif_files():
    # creates directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    pdbs = glob.glob(pdb_dir + "/*.cif")
    dirs = [
        "motifs",
        "motif_interactions",
    ]
    for d in dirs:
        __safe_mkdir(d)
    motif_dir = "motifs/nways/all"
    # opens the file where information about nucleotide interactions are stored
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
    f = open("interactions.csv", "w")
    f.write("name,type,size")
    # writes to the CSV information about nucleotide interactions
    f.write(",".join(hbond_vals) + "\n")
    count = 0
    # CSV about ind. interactions
    f_inter = open("interactions_detailed.csv", "w")
    f_inter.write(
        "name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2" + "\n")

    # CSV listing all th residues present in a given motif
    f_residues = open("motif_residues_list.csv", "w")
    f_residues.write("motif_name,residues" + "\n")

    # CSV for twoway motifs
    f_twoways = open("twoway_motif_list.csv", "w")
    f_twoways.write(
        "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1" + "\n")

    # writes motif/motif interaction information to PDB files
    for pdb_path in pdbs:
        # debug, here we define which exact pdb to run (if we need to for whatever reason)
        # if pdb_path == "/Users/jyesselm/PycharmProjects/rna_motif_library/data/pdbs/7PKQ.cif": # change the part before .cif
        s = os.path.getsize(pdb_path)
        name = pdb_path.split("/")[-1][:-4]
        json_path = settings.LIB_PATH + "/data/dssr_output/" + name + ".json"
        if s < 10000000:  # size-limit on PDB; need more RAM for higher limits; run on a 16 GB machine
            count += 1
            print(count, pdb_path, name)
            pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
            (
                motifs,
                motif_hbonds,
                motif_interactions, hbonds_in_motif
            ) = dssr.get_motifs_from_structure(json_path)

            # hbonds_in_motif is a list, describing all the chain.res ids with hbonds
            # some are counted twice so we need to purify to make it unique

            unique_inter_motifs = list(set(hbonds_in_motif))

            for m in motifs:
                print(m.name)
                spl = m.name.split(".")  # this is the filename
                # don't run if these aren't in the motif name
                if not (spl[0] == "TWOWAY" or spl[0] == "NWAY" or spl[0] == "HAIRPIN" or spl[
                    0] == "HELIX"):
                    continue

                # Writing to interactions.csv
                f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")

                # counting of # of hbond interactions (-base:base)
                if m.name not in motif_hbonds:
                    vals = ["0" for _ in hbond_vals]
                else:
                    vals = [str(motif_hbonds[m.name][x]) for x in hbond_vals]

                f.write(",".join(vals) + "\n")
                # if there are no interactions with the motif then it skips and avoids a crash
                try:
                    interactions = motif_interactions[m.name]
                except KeyError:
                    interactions = None  # or any value you want as a default
                # Writing the residues AND interactions to the CIF files
                dssr.write_res_coords_to_pdb(
                    m.nts_long, interactions, pdb_model,
                    motif_dir + "/" + m.name, unique_inter_motifs, f_inter, f_residues, f_twoways
                )

    f.close()
    f_inter.close()
    f_twoways.close()


# tertiary contact detection (interactions between different motifs)
# basically find if there are any interactions between different motifs
# more than 1 hydrogen bond between different motifs = tertiary contact
def __find_tertiary_contacts():
    # create a CSV file to write tertiary contacts to
    f_tert = open("tertiary_contact_list.csv", "w")
    f_tert.write("motif_1,motif_2,type_1,type_2,res_1,res_2,hairpin_len_1,hairpin_len_2" + "\n")

    # also create a CSV to write non-terts to
    f_single = open("single_motif_inter_list.csv", "w")
    f_single.write("motif,type_1,type_2,res_1,res_2,nt_1,nt_2,distance,angle" + "\n")

    # Specify paths to the CSV files
    interactions_csv_path = "interactions_detailed.csv"
    motif_residues_csv_path = "motif_residues_list.csv"

    # Read the interactions CSV file into a DataFrame and obtain its contents
    interactions_csv_df = pd.read_csv(interactions_csv_path)
    # Group the DataFrame by the 'name' column # and Convert the DataFrameGroupBy back into individual DataFrames
    grouped_interactions_csv_df = interactions_csv_df.groupby('name')

    # Load ALL motif residues into a dictionary, with motif names as keys
    # Open the CSV file
    with open(motif_residues_csv_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)

        # Initialize a dictionary to store data about motifs and their residues
        motif_residues_dict = {}

        # Iterate over each row in the CSV
        for row in csv_reader:
            # Extract the name (first entry in each row)
            name = row[0]

            # Extract the data (rest of the entries in each row)
            data = row[1:]

            # Store the data in the dictionary
            motif_residues_dict[name] = data

    # Iterate over each motif name, as grouped in the dataframe above
    for interaction_group in grouped_interactions_csv_df:
        # interaction_group[0] is the name, [1] is the actual DF
        # HELIX.7PKQ.3.UGC-GCA.0 is the format; now you have the motif name as str
        name_of_source_motif = interaction_group[0]
        name_split = name_of_source_motif.split(".")
        source_motif_type = name_split[0]
        source_motif_cif_id = str(name_split[1])

        # get the residues for the motif of interest; look up in the dictionary
        residues_in_source_motif = motif_residues_dict.get(name_of_source_motif)  # is a list of strings

        # now get the DF with the interaction data in it
        interaction_data_df = interaction_group[1]

        # iterate over each interaction in the motif
        for _, row in interaction_data_df.iterrows():
            # interaction data format: name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle... (don't need the rest)
            # convert datatype series to list
            interaction_data = row.tolist()
            res_1 = interaction_data[1]
            res_2 = interaction_data[2]  # all are strings

            # only for f_single
            type_1 = interaction_data[3]
            type_2 = interaction_data[4]

            if len(type_1) == 1:
                nt_1 = "nt"
            else:
                nt_1 = "aa"

            if len(type_2) == 1:
                nt_2 = "nt"
            else:
                nt_2 = "aa"

            distance_data = str(interaction_data[7])
            angle_data = str(interaction_data[8])

            res_1_present = False
            res_2_present = False

            # need to check if the residue is present in ANY residue list where the CIF id is the same
            # so first we filter to get all the motifs in the same CIF
            dict_with_source_motif_PDB_motifs = {}
            first_line_skipped = False

            # filter mechanism
            for key, value in motif_residues_dict.items():
                if not first_line_skipped:
                    first_line_skipped = True
                    continue

                if key.split('.')[1] == source_motif_cif_id:
                    dict_with_source_motif_PDB_motifs[key] = value

            # if either residue in the interaction is present in the source motif
            if res_1 in residues_in_source_motif:
                res_1_present = True
            if res_2 in residues_in_source_motif:
                res_2_present = True

            # check if residues are present, and if they are, handle them accordingly
            if (res_1_present == False) and (res_2_present == False):
                # not a tert_contact
                # prints interactions - tert contacts to CSV
                f_single.write(
                    name_of_source_motif + "," + type_1 + "," + type_2 + "," + res_1 + "," + res_2 + "," + nt_1 + "," + nt_2 + "," + distance_data + "," + angle_data + "\n")


            elif (res_1_present == True) and (res_2_present == True):
                # not a tert_contact
                pass

            elif (res_1_present == True) and (res_2_present == False):
                # tert contact found
                # res_1 is present in the current motif, res_2 is elsewhere so need to find it
                # now find which motif res_2 is in
                for motif_name, motif_residue_list in dict_with_source_motif_PDB_motifs.items():
                    # Check if the given string is present in the list
                    if res_2 in motif_residue_list:
                        print(motif_name)
                        motif_name_split = motif_name.split(".")
                        motif_name_type = motif_name_split[0]

                        # if the motifs are hairpins, get length
                        if motif_name_type == "HAIRPIN":
                            hairpin_length_1 = str(name_split[2])
                            hairpin_length_2 = str(motif_name_split[2])
                        else:
                            hairpin_length_1 = "0"
                            hairpin_length_2 = "0"

                        # print data to CSV
                        f_tert.write(
                            name_of_source_motif + "," + motif_name + "," + source_motif_type + "," + motif_name_type + "," + res_1 + "," + res_2 + "," + hairpin_length_1 + "," + hairpin_length_2 + "\n")

            elif (res_1_present == False) and (res_2_present == True):
                # tert contact found
                # res_2 is present in the current motif, res_1 is elsewhere
                res_2_data = (res_2, name_of_source_motif)
                # now find which motif res_1 is in
                for motif_name, motif_residue_list in dict_with_source_motif_PDB_motifs.items():
                    # Check if the given string is present in the list
                    if res_1 in motif_residue_list:
                        print(motif_name)
                        motif_name_split = motif_name.split(".")
                        motif_name_type = motif_name_split[0]

                        # if the motifs are hairpins, get length
                        if motif_name_type == "HAIRPIN":
                            hairpin_length_1 = str(name_split[2])
                            hairpin_length_2 = str(motif_name_split[2])
                        else:
                            hairpin_length_1 = "0"
                            hairpin_length_2 = "0"

                        # print data to CSV (only if the motif names are the same)
                        f_tert.write(
                            name_of_source_motif + "," + motif_name + "," + source_motif_type + "," + motif_name_type + "," + res_1 + "," + res_2 + "," + hairpin_length_1 + "," + hairpin_length_2 + "\n")

    # (maybe filter the CSV first to keep those with 2+ interactions)
    # after the CSV for tertiary contacts are made we need to go through and extract all unique pairs in CSV
    # File path
    tert_contact_csv_path = "tertiary_contact_list.csv"
    tert_contact_csv_df = pd.read_csv(tert_contact_csv_path, skiprows=[0])

    # Check if the required columns are present
    required_columns = ['motif_1', 'motif_2']
    if not set(required_columns).issubset(tert_contact_csv_df.columns):
        print(f"A line in the CSV is blank. If this shows only once, it is working as intended.")
        return

    motifs_1 = tert_contact_csv_df['motif_1'].tolist()
    motifs_2 = tert_contact_csv_df['motif_2'].tolist()

    types_1 = tert_contact_csv_df['type_1'].tolist()
    types_2 = tert_contact_csv_df['type_2'].tolist()

    ress_1 = tert_contact_csv_df['res_1'].tolist()
    ress_2 = tert_contact_csv_df['res_2'].tolist()

    # Create a list of tuples
    motif_pairs = [(motif1, motif2, types1, types2, ress1, ress2) for motif1, motif2, types1, types2, ress1, ress2 in
                   zip(motifs_1, motifs_2, types_1, types_2, ress_1, ress_2)]

    # Count occurrences of each unique pair
    pair_counts = Counter(motif_pairs)

    # Create a list of tuples with the third element specifying the count
    unique_motif_pairs_with_count = [
        (pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair_counts[frozenset(pair)]) for pair in
        set(motif_pairs)]

    # Specify the file path
    csv_file_path = "unique_tert_contacts.csv"

    # Open the file for writing
    file = open(csv_file_path, mode='w', newline='')

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["motif_1", "motif_2", "type_1", "type_2", "res_1", "res_2", "count"])

    # Write the unique motif pairs along with their counts
    for pair in unique_motif_pairs_with_count:
        writer.writerow([pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6]])

    # Close the file
    file.close()

    # make directory for tert contacts
    __safe_mkdir("tertiary_contacts")

    # combine the CIFs of tertiary interactions
    for motif_pair in unique_motif_pairs_with_count:
        motif_1 = motif_pair[0]
        motif_2 = motif_pair[1]

        directory_to_search = "motifs"

        motif_cif_1 = str(motif_1) + ".cif"
        motif_cif_2 = str(motif_2) + ".cif"

        path_to_cif_1 = find_cif_file(directory_to_search, motif_cif_1)
        path_to_cif_2 = find_cif_file(directory_to_search, motif_cif_2)

        # path
        tert_contact_name = motif_1 + "." + motif_2
        # debug

        # classifying them based on motif type
        motif_1_type = motif_1.split(".")[0]
        motif_2_type = motif_2.split(".")[0]
        motif_types_list = [motif_1_type, motif_2_type]
        motif_types_sorted = sorted(motif_types_list)

        motif_types = str(motif_types_sorted[0]) + "-" + str(motif_types_sorted[1])

        if motif_types:
            __safe_mkdir("tertiary_contacts/" + motif_types + "/")
            tert_contact_out_path = "tertiary_contacts/" + motif_types + "/" + tert_contact_name

        else:
            tert_contact_out_path = "tertiary_contacts/" + tert_contact_name

        # take the CIF files and merge them
        merge_cif_files(file1_path=path_to_cif_1, file2_path=path_to_cif_2, output_path=f"{tert_contact_out_path}.cif",
                        lines_to_delete=24)
        print(tert_contact_name)


# calculate heatmap for twoway junctions
# function here

def __heatmap_creation():
    # need a CIF of all the twoway junctions to be made upstream
    # motif_name, motif_type (NWAY/TWOWAY), nucleotides_in_strand_1, nucleotides_in_strand_2

    # the classification is done so we just need to import the CSV and build the heatmap
    # Read the CSV data into a DataFrame
    csv_path = "twoway_motif_list.csv"

    try:
        df = pd.read_csv(csv_path)

        # Check if there is any data in the DataFrame
        if df.empty:
            print("No data in the CSV file. Skipping twoway junction processing.")
            return
    except pd.errors.EmptyDataError:
        print(
            "EmptyDataError: No data in the CSV file regarding twoway junctions. Skipping twoway junction processing.")
        return

    # Create a DataFrame for the heatmap
    heatmap_df = df.pivot_table(index='bridging_nts_0', columns='bridging_nts_1', aggfunc='size', fill_value=0)

    # Create a heatmap using seaborn
    sns.heatmap(heatmap_df, cmap='gray_r', fmt='g')

    # Add these lines:
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('scaled')

    # Set the x and y axis labels to numeric scale
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns)
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    # Set the plot labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Twoway Motif Heatmap")

    # Save the heatmap as a PNG file
    plt.savefig("twoway_motif_heatmap.png", dpi=250)

    # Don't display the plot
    plt.close()

    # need to create lists to create a final histogram
    heatmap_res_names = []
    heatmap_atom_names = []

    # TODO heatmaps all the H-bonds
    # we need a heatmap for all the H-bonds present in the data

    # first, take all the h-bonds present in CSV
    hbond_df = pd.read_csv("interactions_detailed.csv")
    # now delete all non-canonical residues
    filtered_hbond_df = hbond_df[
        hbond_df['res_1_name'].isin(canon_res_list) & hbond_df['res_2_name'].isin(canon_res_list)]

    # next, group by (res_1_name, res_2_name) as well as by atoms involved in the interaction
    grouped_hbond_df = filtered_hbond_df.groupby(["res_1_name", "res_2_name", "atom_1", "atom_2"])

    # finally, for each group, make heatmaps of (distance,angle)
    for group in grouped_hbond_df:
        # group[0] = tuple (type_1,type_2), both strings
        # group[1] = dataframe
        group_name = group[0]

        type_1 = str(group_name[0])
        type_2 = str(group_name[1])

        atom_1 = str(group_name[2])
        atom_2 = str(group_name[3])

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        hbonds = group[1]
        # dataframe with the data
        hbonds_subset = hbonds[['distance', 'angle']]
        hbonds_subset = hbonds_subset.reset_index(drop=True)

        if (len(hbonds_subset) >= 100) & (len(hbonds_subset) <= 400):

            # Define the bin intervals for distance and angle
            distance_bins = [i / 4 for i in range(17)]  # Bins from 0 to 4 in increments of 0.25
            angle_bins = [i for i in range(0, 181, 5)]  # Bins from 0 to 180 in increments of 5

            # Bin the data
            hbonds_subset['distance_bin'] = pd.cut(hbonds_subset['distance'], bins=distance_bins)
            hbonds_subset['angle_bin'] = pd.cut(hbonds_subset['angle'], bins=angle_bins)

            # Count the frequency of data points in each bin
            heatmap_data = hbonds_subset.groupby(['angle_bin', 'distance_bin']).size().unstack(fill_value=0)

            # Create the heatmap
            plt.figure(figsize=(6, 10))  # Adjust the figure size as needed
            heatmap = sns.heatmap(heatmap_data, cmap='gray_r', xticklabels=1, yticklabels=range(0, 181, 5), square=True)

            # Set the plot labels and title
            plt.xlabel("Distance (angstroms)")
            plt.ylabel("Angle (degrees)")
            map_name = type_1 + "-" + type_2 + " " + atom_1 + "-" + atom_2
            plt.title(map_name + " H-bond heatmap")

            print(len(type_1))
            print(len(type_2))

            if len(type_1) == 1 and len(type_2) == 1:
                map_dir = "heatmaps/RNA-RNA"
            else:
                map_dir = "heatmaps/RNA-PROT"

            __safe_mkdir(map_dir)

            map_dir = map_dir + "/" + map_name
            # Save the heatmap as a PNG file
            plt.savefig(f"{map_dir}.png", dpi=250)
            # Sometimes the terminal might kill the process
            # if that happens lower the DPI setting above

            # Don't display the plot
            plt.close()

            # Also print a CSV of the appropriate data
            heatmap_csv_path = "heatmap_data"
            __safe_mkdir(heatmap_csv_path)

            # set name
            heat_data_csv_path = heatmap_csv_path + "/" + map_name + ".csv"

            # print data
            hbonds.to_csv(heat_data_csv_path, index=False)

            # need to print a histogram of the number of data points in each heatmap
            # so need to collect this data first

            heatmap_res_names.append(map_name)
            heatmap_atom_names.append(len(hbonds_subset))

        else:
            print(f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient or too many data points.")

    # after collecting data make the final histogram of all the data in heatmaps
    # first compile the list into a df
    histo_df = pd.DataFrame({"heatmap": heatmap_res_names, "count": heatmap_atom_names})

    # plot histogram
    plt.hist(histo_df.iloc[:, 1], bins=400)  # Adjust the number of bins as needed

    # set labels
    plt.xlabel('# of datapoints inside a heatmap')
    plt.ylabel('# of heatmaps with X datapoints')
    plt.title('1d_histogram')

    # set y-axis limit
    #plt.ylim(0, max(df.iloc[:, 1]) * 0.9)

    plt.savefig('1d_histo.png')
    plt.close()

# else:
# print(f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient data points.")


# calculate some final statistics
def __final_statistics():
    motif_directory = "/Users/jyesselm/PycharmProjects/rna_motif_library/rna_motif_library/motifs"

    # Create a dictionary to store counts for each folder
    folder_counts = {}

    # for folder in directory, count numbers:
    try:
        # Iterate over all items in the specified directory
        for item_name in os.listdir(motif_directory):
            item_path = os.path.join(motif_directory, item_name)

            # Check if the current item is a directory
            if os.path.isdir(item_path):
                # Perform your action for each folder
                file_count = count_files_with_extension(item_path, ".cif")

                # Store the count in the dictionary
                folder_counts[item_name] = file_count

        print(folder_counts)
    except Exception as e:
        print(f"Error processing folders in directory '{motif_directory}': {e}")


# merges the contents of CIF files (for concatenation because the old way was trash)
def merge_cif_files(file1_path, file2_path, output_path, lines_to_delete):
    # Read the contents of the first CIF file
    with open(file1_path, 'r') as file1:
        content_file1 = file1.readlines()

    # Read the contents of the second CIF file
    with open(file2_path, 'r') as file2:
        content_file2 = file2.readlines()

    # Delete the first x lines from the second CIF file
    content_file2 = content_file2[lines_to_delete:]

    # Combine the contents of the first and modified second CIF files
    merged_content = content_file1 + content_file2

    # Write the merged content to the output CIF file
    with open(output_path, 'w') as output_file:
        output_file.writelines(merged_content)


# finds CIF files inside the given directory plus all subdirectories
def find_cif_file(directory_path, file_name):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Check if the file is in the current directory
        if file_name in files:
            # Return the full path to the file
            return os.path.join(root, file_name)

    # If the file is not found
    return None


# counts all files with a specific extension
def count_files_with_extension(directory_path, file_extension):
    try:
        # Initialize a counter
        file_count = 0

        # Iterate over the directory and its subdirectories
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                # Check if the current file has the specified extension
                if filename.endswith(file_extension):
                    file_count += 1

        return file_count
    except Exception as e:
        print(f"Error counting files in directory '{directory_path}': {e}")
        return None


def main():
    warnings.filterwarnings("ignore")  # blocks the ragged nested sequence warning
    # time tracking stuff, tracks how long the process takes
    current_time = datetime.datetime.now()
    start_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # start of program

    # the download of a redundant set
    #__download_redundant_cif_files()
    # redundant and nonredundant sets are mutually exclusive and must be run on separate runs
    # with contents cleaned out between runs

    # the download of a nonredundant set
    csv_path = settings.LIB_PATH + "/data/csvs/nrlist_3.320_3.5A.csv"
    #__download_cif_files(csv_path)
    print('''
    ╔════════════════════════════════════╗
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║       CIF FILES DOWNLOADED         ║
    ║                                    ║
    ╚════════════════════════════════════╝
    ''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    #__get_dssr_files()
    print('''
    ╔════════════════════════════════════╗
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║       DSSR FILES FINISHED          ║
    ║                                    ║
    ╚════════════════════════════════════╝
    ''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    #__get_snap_files()
    print('''
    ╔════════════════════════════════════╗
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║       SNAP FILES FINISHED          ║
    ║                                    ║
    ╚════════════════════════════════════╝
    ''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    #__generate_motif_files()
    print('''
    ╔════════════════════════════════════╗
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║      MOTIF FILES FINISHED          ║
    ║                                    ║
    ╚════════════════════════════════════╝
    ''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    #__find_tertiary_contacts()
    print('''
    ╔════════════════════════════════════╗
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║                                    ║
    ║      TERTIARY CONTACTS FINISHED    ║
    ║                                    ║
    ╚════════════════════════════════════╝
        ''')

    ### make a heatmap of the 2way junction data
    print("Printing heatmaps of data...")
    __heatmap_creation()

    print("Final statistics incoming...")
    __final_statistics()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job started on", start_time_string)
    print("Job finished on", time_string)


if __name__ == '__main__':
    main()
