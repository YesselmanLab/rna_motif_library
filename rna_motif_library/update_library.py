import concurrent.futures
import csv
from collections import Counter

import wget
import glob
import os
import datetime
import warnings

from typing import Dict
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import settings
import snap
import dssr
from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS

# from dssr import total_motifs_count
# from dssr import removed_motifs_count

# NOT CODE THINGS just things to remember
# get new real data and analyze
# meet with joe to discuss this thing again soon


# NOT CODE THINGS but things to do nonetheless
# TODO read RNA 3d structure database papers
# TODO start sketching out an introduction
# TODO look into packaging this stuff and getting ready to distribute as a python package

# fix these:
# nothing to fix for now


# questions
# about dihedral angles:
# does directionality matter for our purposes, as in, should I pay attention to +/- 180 degrees whist recording interaction angles?
# I don't think it does, as directionality depends purely on which atom you take first while calculating
# I'm more or less consistent about this, I think as long as we mention it in the methods it should be fine

# done:

# fix misclassification of motif types while finding tertiary contacts
# fix classification of n-way junctions so that it refers to the number of nucleotides in each strand
# fix duplicate motifs showing up in data that is plotted in tertiary contact plots
# put a gap between the bars on graphs and fix graph styles
# check if DA/C/U/G are numerous enough to include as sometimes they are incorrectly counted as nucleosides (they are not; over 36 PDBs I found around 30/5000)
# fix interactions.csv so it includes ALL interactions and properly counts them
# increase text sizes on all figures
# add a column to unique_tert_contacts.csv that records base/sugar/phos data (which part of things are contacts coming from)
# add figure that describes types of tertiary contacts by base/sugar/phos
# deleted "questionable" and "unknown" donAcc type h-bonds
# merged dist-angle data so reverse orders are the same (might need to minus angle by 180 for those) (this one might need a bit of work)
# multithreading (undone, this is slower than single-threading holy fuck)
# added single strands and their respective graphs; make sure tert contacts between others and SSTRAND works


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

# list of residue types to filter out
# may need to include nucleosides and shit
canon_res_list = ['A', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'C', 'G', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'U', 'VAL']

canon_amino_acid_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


# Pandas mmCIF override; need it because of ATOM/HETATM inconsistency

# I hate how there's no universal standard format for CIFs
# Finding these inconsistencies between CIFs took forever
# I'm sure there's still some out there ready to ruin my day that haven't been caught that someone somewhere will catch
# If you see some weird formatting going on in the code, that is the reason
# If you run into issues running this, do fix your CIFs beforehand
# Something like 75% of my time working on this was spent diagnosing and fixing that kind of thing
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


# Download non-redundant set based on CSV
def __download_cif_files(csv_path):
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    count = 0

    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    # Specification of column names
    column_names = ["equivalence_class", "represent", "class_members"]

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


# Runs DSSR on PDBs to extract secondary structure
def __get_dssr_files():
    count = 1
    # creates and sets directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    dssr_path = settings.DSSR_EXE
    out_path = settings.LIB_PATH + "/data/dssr_output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pdbs = glob.glob(pdb_dir + "/*.cif")

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


# Runs snap to get RNP interactions
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
        # s = os.path.getsize(pdb_path)
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
    # pdb_dir = settings.LIB_PATH + "/data/pdbs/"


# old code
"""def __generate_motif_files():
    # defines directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    # rnp_dir = settings.LIB_PATH + "/data/snap_output"
    # grabs all the stuff
    pdbs = glob.glob(pdb_dir + "/*.cif")  # PDBs
    # rnps = glob.glob(rnp_dir + "/*.out")  # RNP interactions
    # creates directories
    dirs = [
        "motifs",
        "motif_interactions",
    ]
    for d in dirs:
        __safe_mkdir(d)
    motif_dir = "motifs/nways/all"
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
    # opens the file where information about nucleotide interactions are stored
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
        name = pdb_path.split("/")[-1][:-4]
        count += 1
        print(count, pdb_path, name)

        # debug, here we define which exact pdb to run (if we need to for whatever reason)
        # if pdb_path == "/Users/jyesselm/PycharmProjects/rna_motif_library/data/pdbs/7PKQ.cif": # change the part before .cif
        s = os.path.getsize(pdb_path)
        json_path = settings.LIB_PATH + "/data/dssr_output/" + name + ".json"
        # if s < 100000000:  # size-limit on PDB; enable if machine runs out of RAM

        # get RNP interactions
        rnp_out_path = settings.LIB_PATH + "/data/snap_output/" + name + ".out"
        # is a list of snap.RNPInteraction objects
        rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)

        # prepare list for RNP data
        rnp_data = []

        for interaction in rnp_interactions:
            # print(interaction)
            atom1, res1 = interaction.nt_atom.split("@")
            atom2, res2 = interaction.aa_atom.split("@")

            # list format: (res1, res2, atom1, atom2, distance)
            rnp_interaction_tuple = (res1, res2, atom1, atom2, str(interaction.dist))
            # rnp_data should be imported into interactions
            rnp_data.append(rnp_interaction_tuple)

        pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
        (
            motifs,
            motif_hbonds,
            motif_interactions, hbonds_in_motif
        ) = dssr.get_motifs_from_structure(json_path)

        # hbonds_in_motif is a list, describing all the chain.res ids with hbonds
        # some are counted twice so we need to purify to make it unique
        # RNP data from snap is injected here
        hbonds_in_motif.extend(rnp_data)
        unique_inter_motifs = list(set(hbonds_in_motif))

        # counting total motifs present in PDB
        dssr.total_motifs_count = len(motifs)
        # process each motif present in PDB
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
                interactions = None  # this means that no interactions are found between the motif and other
            # Writing the residues AND interactions to the CIF files
            dssr.write_res_coords_to_pdb(
                m.nts_long, interactions, pdb_model,
                motif_dir + "/" + m.name, unique_inter_motifs, f_inter, f_residues, f_twoways
            )
        # count motifs found
        # get the PDB name
        ...
        spl_pdbs = pdb_path.split("/")
        pdb_name = spl_pdbs[6].split(".")[0]
        print("Motifs removed:")
        print(dssr.removed_motifs_count)

        if dssr.total_motifs_count != 0:
            print("Percentage removed:")
            percentage_removed = (dssr.removed_motifs_count / dssr.total_motifs_count) * 100
            print(percentage_removed)
        ...

    f.close()
    f_inter.close()
    f_twoways.close()"""


# old new code; not multi threaded
def __generate_motif_files():
    # defines directories
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    # grabs all the stuff
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))  # PDBs
    # creates directories
    dirs = ["motifs"]  # motif_interactions?
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = os.path.join("motifs", "nways", "all")
    hbond_vals = [
        "base:base", "base:sugar", "base:phos", "sugar:base", "sugar:sugar", "sugar:phos",
        "phos:base", "phos:sugar", "phos:phos", "base:aa", "sugar:aa", "phos:aa"
    ]

    # opens the file where information about nucleotide interactions are stored
    with open("interactions.csv", "w") as f_inter_overview:
        f_inter_overview.write("name,type,size," + ",".join(hbond_vals) + "\n")
        # CSV about individual interactions
        with open("interactions_detailed.csv", "w") as f_inter, \
                open("motif_residues_list.csv", "w") as f_residues, \
                open("twoway_motif_list.csv", "w") as f_twoways:

            f_inter.write(
                "name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2\n")
            f_residues.write("motif_name,residues\n")
            f_twoways.write(
                "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1\n")

            count = 0
            for pdb_path in pdbs:
                # limit so we can test some stuff at a smaller scale
                # if count > 50:
                #    continue
                name = os.path.basename(pdb_path).replace(".cif", "")
                count += 1
                print(count, pdb_path, name)

                s = os.path.getsize(pdb_path)
                json_path = os.path.join(settings.LIB_PATH, "data/dssr_output", name + ".json")
                rnp_out_path = os.path.join(settings.LIB_PATH, "data/snap_output", name + ".out")
                rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)
                rnp_data = [(interaction.nt_atom.split("@")[1], interaction.aa_atom.split("@")[1],
                             interaction.nt_atom.split("@")[0], interaction.aa_atom.split("@")[0],
                             str(interaction.dist), interaction.type.split(":")[0], interaction.type.split(":")[1]) for
                            interaction in rnp_interactions]

                pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
                motifs, motif_hbonds, motif_interactions, hbonds_in_motif = dssr.get_motifs_from_structure(json_path)

                # add RNP data
                hbonds_in_motif.extend(rnp_data)
                unique_inter_motifs = list(set(hbonds_in_motif))
                # format of elements in unique_inter_motifs:
                # ('A.ASP126', 'A.GNP402', 'OD2', 'N2', '2.824', 'base', 'aa')

                dssr.total_motifs_count = len(motifs)
                for m in motifs:
                    # print(m.name)
                    spl = m.name.split(".")
                    if spl[0] not in ["TWOWAY", "NWAY", "HAIRPIN", "HELIX", "SSTRAND"]:
                        continue

                    interactions = motif_interactions.get(m.name, None)
                    dssr.write_res_coords_to_pdb(m.nts_long, interactions, pdb_model,
                                                 os.path.join(motif_dir, m.name),
                                                 unique_inter_motifs, f_inter, f_residues, f_twoways, f_inter_overview)

                    """# writing to interactions.csv
                    f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")
                    vals = [str(motif_hbonds[m.name][x]) if m.name in motif_hbonds else "0" for x in hbond_vals]
                    f.write(",".join(vals) + "\n")"""


# new code; multi threaded; multi threads per PDB
"""def __generate_motif_files():
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))
    dirs = ["motifs", "motif_interactions"]
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = os.path.join("motifs", "nways", "all")
    hbond_vals = [
        "base:base", "base:sugar", "base:phos", "sugar:base", "sugar:sugar", "sugar:phos",
        "phos:base", "phos:sugar", "phos:phos", "base:aa", "sugar:aa", "phos:aa"
    ]

    # opens the file where information about nucleotide interactions are stored
    with open("interactions.csv", "w") as f:
        f.write("name,type,size," + ",".join(hbond_vals) + "\n")
        # CSV about individual interactions
        with open("interactions_detailed.csv", "w") as f_inter, \
                open("motif_residues_list.csv", "w") as f_residues, \
                open("twoway_motif_list.csv", "w") as f_twoways:

            f_inter.write("name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2\n")
            f_residues.write("motif_name,residues\n")
            f_twoways.write(
                "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1\n")

            # Use ThreadPoolExecutor for multithreading
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                count = 0
                for pdb_path in pdbs:
                    count += 1
                    future = executor.submit(process_pdb, pdb_path, motif_dir, hbond_vals, count)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        results = future.result()
                        for row, nts_long, interactions, unique_inter_motifs, pdb_model in results:
                            f.write(",".join(row) + "\n")
                            dssr.write_res_coords_to_pdb(nts_long, interactions, pdb_model,
                                                         os.path.join(motif_dir, row[0]),
                                                         unique_inter_motifs, f_inter, f_residues, f_twoways)
                    except Exception as exc:
                        print(f'Exception processing {future}: {exc}')"""

# new code; multi threads per motif
"""def __generate_motif_files():
    # defines directories
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    # grabs all the stuff
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))  # PDBs
    # creates directories
    dirs = ["motifs", "motif_interactions"]
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = os.path.join("motifs", "nways", "all")
    hbond_vals = [
        "base:base", "base:sugar", "base:phos", "sugar:base", "sugar:sugar", "sugar:phos",
        "phos:base", "phos:sugar", "phos:phos", "base:aa", "sugar:aa", "phos:aa"
    ]

    # opens the file where information about nucleotide interactions are stored
    with open("interactions.csv", "w") as f:
        f.write("name,type,size," + ",".join(hbond_vals) + "\n")
        # CSV about individual interactions
        with open("interactions_detailed.csv", "w") as f_inter, \
                open("motif_residues_list.csv", "w") as f_residues, \
                open("twoway_motif_list.csv", "w") as f_twoways:

            f_inter.write("name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2\n")
            f_residues.write("motif_name,residues\n")
            f_twoways.write(
                "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1\n")

            count = 0
            for pdb_path in pdbs:
                name = os.path.basename(pdb_path).replace(".cif", "")
                count += 1
                print(count, pdb_path, name)

                s = os.path.getsize(pdb_path)
                json_path = os.path.join(settings.LIB_PATH, "data/dssr_output", name + ".json")

                rnp_out_path = os.path.join(settings.LIB_PATH, "data/snap_output", name + ".out")
                rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)
                rnp_data = [(interaction.nt_atom.split("@")[1], interaction.aa_atom.split("@")[1],
                             interaction.nt_atom.split("@")[0], interaction.aa_atom.split("@")[0],
                             str(interaction.dist)) for interaction in rnp_interactions]

                pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
                motifs, motif_hbonds, motif_interactions, hbonds_in_motif = dssr.get_motifs_from_structure(json_path)

                hbonds_in_motif.extend(rnp_data)
                unique_inter_motifs = list(set(hbonds_in_motif))

                dssr.total_motifs_count = len(motifs)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for m in motifs:
                        spl = m.name.split(".")
                        if spl[0] not in ["TWOWAY", "NWAY", "HAIRPIN", "HELIX", "SSTRAND"]:
                            continue

                        future = executor.submit(process_motif, m, motif_hbonds, motif_interactions,
                                                 unique_inter_motifs, hbond_vals, pdb_model, motif_dir, f_inter,
                                                 f_residues, f_twoways)
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            row = future.result()
                            f.write(",".join(row) + "\n")
                        except Exception as exc:
                            print(f'Exception processing motif: {exc}')"""

"""def process_motif(m, motif_hbonds, motif_interactions, unique_inter_motifs, hbond_vals, pdb_model, motif_dir, f_inter,
                  f_residues, f_twoways):
    print(m.name)
    row = [m.name, m.name.split(".")[0], str(len(m.nts_long))]
    vals = [str(motif_hbonds[m.name][x]) if m.name in motif_hbonds else "0" for x in hbond_vals]
    row.extend(vals)

    interactions = motif_interactions.get(m.name, None)
    dssr.write_res_coords_to_pdb(m.nts_long, interactions, pdb_model,
                                 os.path.join(motif_dir, m.name),
                                 unique_inter_motifs, f_inter, f_residues, f_twoways)

    return row


def process_pdb(pdb_path, motif_dir, hbond_vals, count):
    name = os.path.basename(pdb_path).replace(".cif", "")
    print(count, pdb_path, name)

    json_path = os.path.join(settings.LIB_PATH, "data/dssr_output", name + ".json")

    rnp_out_path = os.path.join(settings.LIB_PATH, "data/snap_output", name + ".out")
    rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)

    rnp_data = [(interaction.nt_atom.split("@")[1], interaction.aa_atom.split("@")[1],
                 interaction.nt_atom.split("@")[0], interaction.aa_atom.split("@")[0],
                 str(interaction.dist), interaction.type.split(":")[0], interaction.type.split(":")[1]) for interaction in rnp_interactions]

    # rnp_data is the same format as other interactions

    pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
    motifs, motif_hbonds, motif_interactions, hbonds_in_motif = dssr.get_motifs_from_structure(json_path)

    hbonds_in_motif.extend(rnp_data)
    unique_inter_motifs = list(set(hbonds_in_motif))

    print(unique_inter_motifs)
    exit(0)

    results = []

    for m in motifs:
        print(m.name)
        spl = m.name.split(".")
        if spl[0] not in ["TWOWAY", "NWAY", "HAIRPIN", "HELIX", "SSTRAND"]:
            continue

        row = [m.name, spl[0], str(len(m.nts_long))]
        vals = [str(motif_hbonds[m.name][x]) if m.name in motif_hbonds else "0" for x in hbond_vals]
        row.extend(vals)

        # Process each motif concurrently
        f_inter = open("interactions_detailed.csv", "a")  # Append mode for concurrent writing
        f_residues = open("motif_residues_list.csv", "a")  # Append mode for concurrent writing
        f_twoways = open("twoway_motif_list.csv", "a")  # Append mode for concurrent writing

        interactions = motif_interactions.get(m.name, None)
        dssr.write_res_coords_to_pdb(m.nts_long, interactions, pdb_model,
                                     os.path.join(motif_dir, m.name),
                                     unique_inter_motifs, f_inter, f_residues, f_twoways)

        f_inter.close()
        f_residues.close()
        f_twoways.close()

        results.append((row, m.nts_long, interactions, unique_inter_motifs, pdb_model))

    return results"""


# tertiary contact detection (interactions between different motifs)
# basically find if there are any interactions between different motifs
# more than 1 hydrogen bond between different motifs = tertiary contact
def __find_tertiary_contacts():
    # TODO finding starts here, comment out if found already or need to debug, not actual todo
    print("Finding tertiary contacts...")

    # create a CSV file to write tertiary contacts to
    f_tert = open("tertiary_contact_list.csv", "w")
    f_tert.write("motif_1,motif_2,type_1,type_2,res_1,res_2,hairpin_len_1,hairpin_len_2,res_type_1,res_type_2" + "\n")

    # also create a CSV to write non-tert contacts to
    f_single = open("single_motif_inter_list.csv", "w")
    f_single.write("motif,type_1,type_2,res_1,res_2,nt_1,nt_2,distance,angle,res_type_1,res_type_2" + "\n")

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
            # interaction data format: name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2
            # convert datatype series to list
            interaction_data = row.tolist()
            res_1 = interaction_data[1]
            res_2 = interaction_data[2]  # all are strings

            # only for f_single
            type_1 = interaction_data[3]
            type_2 = interaction_data[4]

            res_type_1 = interaction_data[11]
            res_type_2 = interaction_data[12]

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
                    name_of_source_motif + "," + type_1 + "," + type_2 + "," + res_1 + "," + res_2 + "," + nt_1 + "," + nt_2 + "," + distance_data + "," + angle_data + ',' + res_type_1 + ',' + res_type_2 + "\n")


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
                            name_of_source_motif + "," + motif_name + "," + source_motif_type + "," + motif_name_type + "," + res_1 + "," + res_2 + "," + hairpin_length_1 + "," + hairpin_length_2 + "," + res_type_1 + "," + res_type_2 + "\n")

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
                            name_of_source_motif + "," + motif_name + "," + source_motif_type + "," + motif_name_type + "," + res_1 + "," + res_2 + "," + hairpin_length_1 + "," + hairpin_length_2 + "," + res_type_1 + "," + res_type_2 + "\n")

    # TODO finding ends here; not actual todo
    # after the CSV for tertiary contacts are made we need to go through and extract all unique pairs in CSV
    # File path
    tert_contact_csv_path = "tertiary_contact_list.csv"  # used to make unique list which then is used for everything else
    tert_contact_csv_df = pd.read_csv(tert_contact_csv_path)

    # Check if the required columns are present
    required_columns = ['motif_1', 'motif_2']
    if not set(required_columns).issubset(tert_contact_csv_df.columns):
        print(f"A line in the CSV is blank. If this shows only once, it is working as intended.")
        # return # return if you don't want to print tert contacts

    # TODO unique tert contact discovery starts here; not actual todo

    print("Finding unique tertiary contacts...")
    # Count unique tert contacts and print to a CSV
    grouped_unique_tert_contacts = tert_contact_csv_df.groupby(['motif_1', 'motif_2', 'res_1', 'res_2'])

    # Specify the file path
    # Making the unique_tert_contacts.csv file
    print("Opened unique_tert_contacts.csv")
    csv_file_path = "unique_tert_contacts.csv"
    print("Writing to unique_tert_contacts.csv...")
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write the header
        writer.writerow(
            ["motif_1", "motif_2", "seq_1", "seq_2", "type_1", "type_2", "res_1", "res_2", "count", "res_type_1",
             "res_type_2"])

        # Iterate over groups
        for group_name, group_df in grouped_unique_tert_contacts:
            # Get the count of rows in the group
            count = len(group_df)

            # Drop duplicates
            group_df_unique = group_df.drop_duplicates(keep='first')

            # Iterate over rows to write to CSV
            for index, row in group_df_unique.iterrows():
                # Split motif_1 and motif_2 by "." and take all parts except the last one
                seq_1 = ".".join(row['motif_1'].split(".")[:-1])
                seq_2 = ".".join(row['motif_2'].split(".")[:-1])

                # need to write the motif type correctly or you get fuckups later (sometimes they get switched up)
                motif_1_spl = row['motif_1'].split(".")
                motif_2_spl = row['motif_2'].split(".")

                motif_1_type = motif_1_spl[0]
                motif_2_type = motif_2_spl[0]

                # Write to CSV, appending the count at the end
                writer.writerow(
                    [row['motif_1'], row['motif_2'], seq_1, seq_2, motif_1_type, motif_2_type, row['res_1'],
                     row['res_2'], count, row['res_type_1'], row['res_type_2']])

    # Close the file
    file.close()
    # graph hydrogen bonds per overall tertiary contact
    # Read the CSV file into a DataFrame
    unique_tert_contact_df_unfiltered = pd.read_csv(csv_file_path)
    # Create an empty DataFrame to store the filtered data
    filtered_data = []

    # discount the tert contact if hairpins < 3
    print("Discounting hairpins < 3...")
    # Iterate through each row in the unfiltered DataFrame
    for index, row in unique_tert_contact_df_unfiltered.iterrows():
        # Split motif_1 and motif_2 by "."
        motif_1_split = row['motif_1'].split('.')
        motif_2_split = row['motif_2'].split('.')

        # Check conditions for deletion
        if motif_1_split[0] == 'HAIRPIN' and 0 < float(motif_1_split[2]) < 3:
            continue
        elif motif_2_split[0] == 'HAIRPIN' and 0 < float(motif_2_split[2]) < 3:
            continue
        else:
            # Keep the row by appending it to the filtered_data list
            filtered_data.append(row)

    # Create a new DataFrame with the filtered data
    unique_tert_contact_df_new = pd.DataFrame(filtered_data)

    # Reset the index of the new DataFrame
    unique_tert_contact_df_new.reset_index(drop=True, inplace=True)

    # debug
    # unique_tert_contact_df_new.to_csv("terts_with_duplicates.csv", index=False)

    print("Deleting duplicates...")
    # Now delete duplicate interactions (where motif_1 and 2 are switched)

    # Sort the 'motif_1' and 'motif_2' columns within each row
    unique_tert_contact_df_new[['motif_1', 'motif_2']] = pd.DataFrame(
        np.sort(unique_tert_contact_df_new[['motif_1', 'motif_2']], axis=1), index=unique_tert_contact_df_new.index)

    # Group the DataFrame by the sorted 'motif_1' and 'motif_2' columns
    grouped_df = unique_tert_contact_df_new.groupby(['motif_1', 'motif_2'])

    # Define a function to remove duplicate rows within each group based on sorted 'res_1' and 'res_2' values
    def remove_duplicate_res(group):
        # Sort 'res_1' and 'res_2' columns within each row
        group[['res_1', 'res_2']] = group[['res_1', 'res_2']].apply(sort_res, axis=1)
        # Drop duplicate rows based on sorted 'res_1' and 'res_2' values
        return group.drop_duplicates(subset=['res_1', 'res_2'], keep='first')

    # Apply the function to remove duplicate rows within each group
    unique_tert_contact_df_for_hbonds = grouped_df.apply(remove_duplicate_res).reset_index(drop=True)
    unique_tert_contact_df_for_hbonds.to_csv("unique_tert_contacts_for_hbonds.csv", index=False)

    # If there are fewer than two residues interacting in the contact, delete
    grouped_df = unique_tert_contact_df_for_hbonds.groupby(['motif_1', 'motif_2'])
    # Filter out groups with less than 2 rows
    unique_tert_contact_df_for_hbonds = grouped_df.filter(lambda x: len(x) >= 2)

    # Print it for good measure, debug
    # unique_tert_contact_df_for_hbonds.to_csv("unique_tert_contacts.csv", index=False)

    # Process each group to sum the 'count' column and replace values
    final_data = []
    grouped_df = unique_tert_contact_df_for_hbonds.groupby(['motif_1', 'motif_2'])
    for name, group in grouped_df:
        # Sum the 'count' column within the group
        count_sum = group['count'].astype(int).sum()

        # Set the 'count' column to the summed value
        group['count'] = count_sum

        # Keep only the first row of each group
        final_data.append(group.iloc[0])

    # Create a new DataFrame from the processed data
    unique_tert_contact_df = pd.DataFrame(final_data)

    # Remove duplicates
    # Group the DataFrame by the sorted 'seq_1' and 'seq_2' columns
    grouped_df = unique_tert_contact_df.groupby(['seq_1', 'seq_2'])

    # get rid of duplicate sequences
    unique_tert_contact_df = grouped_df.first().reset_index()

    # Reset the index of the DataFrame
    unique_tert_contact_df.reset_index(drop=True, inplace=True)

    # Print it for good measure
    unique_tert_contact_df.to_csv("unique_tert_contacts.csv", index=False)
    # TODO this one should be used for everything, not actual todo

    # make directory for tert contacts
    __safe_mkdir("tertiary_contacts")

    # TODO combine the CIFs of tertiary interactions and save them to CIF file; not actual todo

    print("Saving tertiary contacts to CIF files...")
    # for printing the tert contact CIFs need to prepare data
    motifs_1 = unique_tert_contact_df['motif_1'].tolist()
    motifs_2 = unique_tert_contact_df['motif_2'].tolist()
    types_1 = unique_tert_contact_df['type_1'].tolist()
    types_2 = unique_tert_contact_df['type_2'].tolist()
    ress_1 = unique_tert_contact_df['res_1'].tolist()
    ress_2 = unique_tert_contact_df['res_2'].tolist()
    # Create a list of tuples
    motif_pairs = [(motif1, motif2, types1, types2, ress1, ress2) for motif1, motif2, types1, types2, ress1, ress2 in
                   zip(motifs_1, motifs_2, types_1, types_2, ress_1, ress_2)]
    # Create a list of tuples with the third element specifying the count
    unique_motif_pairs_with_count = [
        (pair[0], pair[1], pair[2], pair[3], pair[4], pair[5]) for pair in
        set(motif_pairs)]
    for motif_pair in unique_motif_pairs_with_count:
        motif_1 = motif_pair[0]
        motif_2 = motif_pair[1]
        # if motif_1 or motif_2 are hairpins less than 3 NTS long then don't process
        motif_1_split = motif_1.split(".")
        motif_2_split = motif_2.split(".")
        motif_1_name = motif_1_split[0]
        motif_2_name = motif_2_split[0]
        try:
            motif_1_hairpin_len = float(motif_1_split[2])
            motif_2_hairpin_len = float(motif_2_split[2])
        except ValueError:
            motif_1_hairpin_len = 0
            motif_2_hairpin_len = 0
        if not ((motif_1_name == "HAIRPIN" or motif_2_name == "HAIRPIN") and (
                (0 < motif_1_hairpin_len < 3) or (0 < motif_2_hairpin_len < 3))):
            directory_to_search = "motifs"
            motif_cif_1 = str(motif_1) + ".cif"
            motif_cif_2 = str(motif_2) + ".cif"
            path_to_cif_1 = find_cif_file(directory_to_search, motif_cif_1)
            path_to_cif_2 = find_cif_file(directory_to_search, motif_cif_2)
            # path
            tert_contact_name = motif_1 + "." + motif_2
            # classifying them based on motif type
            motif_1_type = motif_1.split(".")[0]
            motif_2_type = motif_2.split(".")[0]
            motif_types_list = [motif_1_type, motif_2_type]
            motif_types_sorted = sorted(motif_types_list)
            motif_types = str(motif_types_sorted[0]) + "-" + str(motif_types_sorted[1])

            if motif_types:
                __safe_mkdir("tertiary_contacts/" + motif_types)
                tert_contact_out_path = "tertiary_contacts/" + motif_types + "/" + tert_contact_name
            else:
                tert_contact_out_path = "tertiary_contacts/" + tert_contact_name

            print(tert_contact_name)
            # take the CIF files and merge them
            try:
                merge_cif_files(file1_path=path_to_cif_1, file2_path=path_to_cif_2,
                                output_path=f"{tert_contact_out_path}.cif",
                                lines_to_delete=24)
            except TypeError:
                continue

        else:
            continue

    # TODO plotting starts here; not actual todo
    print("Plotting...")
    # Group by motif_1 and motif_2 and sum the counts
    hbond_counts_in_terts = unique_tert_contact_df_for_hbonds.groupby(['motif_1', 'motif_2'])[
        'count'].sum().reset_index()

    # debug
    # hbond_counts_in_terts.to_csv("hbond_counts_in_terts.csv", index=False)

    # Rename the 'count' column to 'sum_hbonds'
    hbond_counts_in_terts.rename(columns={'count': 'sum_hbonds'}, inplace=True)

    # Remove duplicate lines based on motif_1 and motif_2 columns
    hbond_counts_in_terts.drop_duplicates(subset=['motif_1', 'motif_2'], inplace=True)

    tick_positions = np.arange(hbond_counts_in_terts['sum_hbonds'].min(), hbond_counts_in_terts['sum_hbonds'].max() + 1)

    # Now make a histogram
    # Plot histogram
    # H-bonds per tert, need to group the ones with like motifs and sum the tert contacts
    sns.set_theme(style="white")  # palette='deep', color_codes=True)
    plt.rcParams.update({'font.size': 20})  # Set overall text size

    plt.figure(figsize=(6, 6))
    plt.hist(hbond_counts_in_terts['sum_hbonds'],
             bins=np.arange(hbond_counts_in_terts['sum_hbonds'].min() - 0.5,
                            hbond_counts_in_terts['sum_hbonds'].max() + 1.5, 1),
             edgecolor='black', width=0.8)  # adjust bins as needed
    plt.xlabel('H-bonds per tertiary contact')
    plt.ylabel('Count')
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5)
    # Save the plot as PNG file
    plt.savefig('figure_3_hbonds_per_tert.png', dpi=600)
    # Close the plot
    plt.close()
    # Now make a histogram for lengths of hairpins in tertiary contacts
    # split into two DFs
    df_cols_1 = ['motif_1', 'type_1', 'res_1', 'seq_1']
    tert_contact_df_1 = unique_tert_contact_df[df_cols_1]
    df_cols_2 = ['motif_2', 'type_2', 'res_2', 'seq_2']
    tert_contact_df_2 = unique_tert_contact_df[df_cols_2]
    # Filter rows where hairpins_1 and hairpins_2 are equal to "HAIRPIN"
    tert_contact_df_1 = tert_contact_df_1[tert_contact_df_1['type_1'] == "HAIRPIN"]
    tert_contact_df_2 = tert_contact_df_2[tert_contact_df_2['type_2'] == "HAIRPIN"]
    # split
    split_column_1 = tert_contact_df_1['motif_1'].str.split('.')
    split_column_2 = tert_contact_df_2['motif_2'].str.split('.')
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    tert_contact_df_1 = tert_contact_df_1.assign(length_1=length_1)
    tert_contact_df_2 = tert_contact_df_1.assign(length_2=length_2)
    # Concatenate tert_contact_df_1 and tert_contact_df_2
    new_tert_df = pd.concat([tert_contact_df_1, tert_contact_df_2], ignore_index=True, axis=0)
    # List of column names to delete (duplicates, since the were concatenated one on top of another)
    columns_to_delete = ['length_2']
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    # And delete dupes
    new_tert_df.drop_duplicates(subset=['seq_1'], keep='first', inplace=True)
    # Rename columns of tert_contact_df_1
    new_tert_df.columns = ['motif', 'type', 'res', 'seq', 'hairpin_length']
    for index, row in new_tert_df.iterrows():
        seq_value = row['seq']
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            hairpin_length_value = int(parts[2])
            new_tert_df.at[index, 'hairpin_length'] = hairpin_length_value

    # Print for debug reasons
    new_tert_df.to_csv("hairpins_tert.csv", index=False)
    # Convert data to numeric from string
    new_tert_df['hairpin_length'] = pd.to_numeric(new_tert_df['hairpin_length'], errors='coerce')
    # Set tick positions to fit the range of data
    tick_positions = np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1)
    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(new_tert_df['hairpin_length'],
             bins=np.arange(new_tert_df['hairpin_length'].min() - 0.5, new_tert_df['hairpin_length'].max() + 1.5, 1),
             edgecolor='black', width=0.8)  # adjust bins as needed
    plt.xlabel('Length of hairpins in tertiary contacts')
    plt.ylabel('Count')
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5); old code
    # Save the plot as PNG file
    plt.savefig('figure_3_hairpins_in_tert.png', dpi=600)
    # Close the plot
    plt.close()

    # helices in tertiary contacts
    # filter to get only helices
    helix_cols_1 = ['motif_1', 'type_1', 'res_1', 'seq_1']
    helix_tert_contact_df_1 = unique_tert_contact_df[helix_cols_1]
    helix_cols_2 = ['motif_2', 'type_2', 'res_2', 'seq_2']
    helix_tert_contact_df_2 = unique_tert_contact_df[helix_cols_2]
    # Filter rows where types are equal to "HELIX"
    helix_tert_contact_df_1 = helix_tert_contact_df_1[helix_tert_contact_df_1['type_1'] == "HELIX"]
    helix_tert_contact_df_2 = helix_tert_contact_df_2[helix_tert_contact_df_2['type_2'] == "HELIX"]
    # split
    split_column_1 = helix_tert_contact_df_1['motif_1'].str.split('.')
    split_column_2 = helix_tert_contact_df_2['motif_2'].str.split('.')
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    helix_tert_contact_df_1 = helix_tert_contact_df_1.assign(length_1=length_1)
    helix_tert_contact_df_2 = helix_tert_contact_df_2.assign(length_2=length_2)
    # concatenate and get rid of dupes
    new_tert_df = pd.concat([helix_tert_contact_df_1, helix_tert_contact_df_2], ignore_index=True, axis=0)
    new_tert_df.drop_duplicates(subset=['seq_1'], keep='first', inplace=True)
    # List of column names to delete
    columns_to_delete = ['motif_2', 'type_2', 'res_2', 'seq_2', 'length_2']
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    new_tert_df.columns = ['motif', 'type', 'res', 'seq', 'helix_length']

    for index, row in new_tert_df.iterrows():
        seq_value = row['seq']
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            helix_length_value = int(parts[2])
            new_tert_df.at[index, 'helix_length'] = helix_length_value

    # Print for debug
    new_tert_df.to_csv("helices_tert.csv", index=False)
    # Convert 'helix_length' column to numeric type
    new_tert_df['helix_length'] = pd.to_numeric(new_tert_df['helix_length'], errors='coerce')
    tick_positions = np.arange(new_tert_df['helix_length'].min(), new_tert_df['helix_length'].max() + 1)
    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(new_tert_df['helix_length'],
             bins=np.arange(new_tert_df['helix_length'].min() - 0.5, new_tert_df['helix_length'].max() + 1.5, 1),
             edgecolor='black', width=0.8)  # adjust bins as needed
    plt.xlabel('Length of helices in tertiary contacts')
    plt.ylabel('Count')
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5)

    # Save the plot as PNG file
    plt.savefig('figure_3_helices_in_tert.png', dpi=600)
    # Close the plot
    plt.close()

    # sstrands in tertiary contacts
    # filter to get only sstrands
    sstrand_cols_1 = ['motif_1', 'type_1', 'res_1', 'seq_1']
    sstrand_tert_contact_df_1 = unique_tert_contact_df[sstrand_cols_1]
    sstrand_cols_2 = ['motif_2', 'type_2', 'res_2', 'seq_2']
    sstrand_tert_contact_df_2 = unique_tert_contact_df[sstrand_cols_2]
    # Filter rows where types are equal to "SSTRAND"
    sstrand_tert_contact_df_1 = sstrand_tert_contact_df_1[sstrand_tert_contact_df_1['type_1'] == "SSTRAND"]
    sstrand_tert_contact_df_2 = sstrand_tert_contact_df_2[sstrand_tert_contact_df_2['type_2'] == "SSTRAND"]
    # split
    split_column_1 = sstrand_tert_contact_df_1['motif_1'].str.split('.')
    split_column_2 = sstrand_tert_contact_df_2['motif_2'].str.split('.')
    # extract length
    length_1 = split_column_1.str[2]
    length_2 = split_column_2.str[2]
    sstrand_tert_contact_df_1 = sstrand_tert_contact_df_1.assign(length_1=length_1)
    sstrand_tert_contact_df_2 = sstrand_tert_contact_df_2.assign(length_2=length_2)
    # Concatenate and drop dupes
    new_tert_df = pd.concat([sstrand_tert_contact_df_1, sstrand_tert_contact_df_2], ignore_index=True, axis=0)
    new_tert_df.drop_duplicates(subset=['seq_1'], keep='first', inplace=True)
    # List of column names to delete
    columns_to_delete = ['motif_2', 'type_2', 'res_2', 'seq_2', 'length_2']
    # Delete the specified columns
    new_tert_df.drop(columns=columns_to_delete, inplace=True)
    # rename columns
    new_tert_df.columns = ['motif', 'type', 'res', 'seq', 'sstrand_length']
    for index, row in new_tert_df.iterrows():
        seq_value = row['seq']
        if isinstance(seq_value, float):
            seq_value = str(seq_value)
        parts = seq_value.split(".")
        if len(parts) > 2:
            sstrand_length_value = int(parts[2])
            new_tert_df.at[index, 'sstrand_length'] = sstrand_length_value

    # Print for debug
    new_tert_df.to_csv("sstrand_tert.csv", index=False)
    # Convert 'sstrand_length' column to numeric type
    new_tert_df['sstrand_length'] = pd.to_numeric(new_tert_df['sstrand_length'], errors='coerce')
    tick_positions = np.arange(new_tert_df['sstrand_length'].min(), new_tert_df['sstrand_length'].max() + 1)

    # Now make a histogram
    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(new_tert_df['sstrand_length'],
             bins=np.arange(new_tert_df['sstrand_length'].min() - 0.5, new_tert_df['sstrand_length'].max() + 1.5, 1),
             edgecolor='black', width=0.8)  # adjust bins as needed
    plt.xlabel('Length of sstrands in tertiary contacts')
    plt.ylabel('Count')
    # Add tick marks on x-axis
    plt.xticks(tick_positions[::5], [int(tick) for tick in tick_positions[::5]])
    # plt.xticks(np.arange(new_tert_df['hairpin_length'].min(), new_tert_df['hairpin_length'].max() + 1), 5)

    # Save the plot as PNG file
    plt.savefig('figure_3_sstrand_in_tert.png', dpi=600)
    # Close the plot
    plt.close()


# calculate heatmap for twoway junctions
# function here

def __heatmap_creation():
    print("Plotting heatmaps...")
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

    # make a heatmap for TWOWAY JCTs
    # Create a DataFrame for the heatmap
    df['bridging_nts_0'] = df['bridging_nts_0'] - 2
    df['bridging_nts_1'] = df['bridging_nts_1'] - 2
    twoway_heatmap_df = df.pivot_table(index='bridging_nts_0', columns='bridging_nts_1', aggfunc='size', fill_value=0)

    # Extract the data from the DataFrame
    x = twoway_heatmap_df.columns.astype(float)
    y = twoway_heatmap_df.index.astype(float)
    z = twoway_heatmap_df.values

    # Reshape the data for hist2d
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Determine the range of x and y
    x_range = np.arange(int(x.min()), min(int(x.max()) + 1, 12))  # Limit to 10 on x-axis
    y_range = np.arange(int(y.min()), min(int(y.max()) + 1, 12))  # Limit to 10 on y-axis

    # Create the 2D histogram
    sns.set_theme(style="white")
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 20})
    heatmap = plt.hist2d(x_mesh.ravel(), y_mesh.ravel(), weights=z.ravel(), bins=[x_range, y_range], cmap='gray_r')

    # Add labels and title
    plt.xlabel("Strand 1 Nucleotides")
    plt.ylabel("Strand 2 Nucleotides")
    # plt.title("Figure 2(e): 2-way junctions (X-Y)", fontsize=32)

    # Add colorbar for frequency scale
    # cbar = plt.colorbar(label='Frequency')

    # Set aspect ratio of color bar to match the height of the plot
    # cbar.ax.set_aspect(40)

    # Set ticks on x-axis
    plt.xticks(np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1),
               [f'{int(tick - 0.5)}' for tick in np.arange(x_range.min() + 0.5, x_range.max() + 1.5, 1)])

    # Set ticks on y-axis
    plt.yticks(np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1),
               [f'{int(tick - 0.5)}' for tick in np.arange(y_range.min() + 0.5, y_range.max() + 1.5, 1)])

    # Set aspect ratio to square
    plt.gca().set_aspect('equal', adjustable='box')

    # Add colorbar for frequency scale
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(heatmap[3], cax=cax)
    cbar.set_label('Count')
    # cbar.ax.tick_params(labelsize=18)

    # Save the heatmap as a PNG file
    plt.savefig("figure_2_twoway_motif_heatmap.png", dpi=600)

    # Don't display the plot
    plt.close()

    # TODO need to create lists to create a final histogram for h-bonds; not actual todo
    # this is for the debugging histogram
    # heatmap_res_names = []
    # heatmap_atom_names = []

    # first, take all the h-bonds present in CSV
    hbond_df_unfiltered = pd.read_csv("interactions_detailed.csv")

    # also delete res_1_name and res_2_name where they are hairpins less than 3
    # Create an empty DataFrame to store the filtered data
    filtered_data = []
    # Iterate through each row in the unfiltered DataFrame
    for index, row in hbond_df_unfiltered.iterrows():
        # Split motif_1 and motif_2 by "."
        motif_1_split = row['name'].split('.')

        # Check conditions for deletion
        if motif_1_split[0] == 'HAIRPIN' and 0 < float(motif_1_split[2]) < 3:
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
        hbond_df['res_1_name'].isin(canon_res_list) & hbond_df['res_2_name'].isin(canon_res_list)]

    # reverse orders are sorted into the same group
    filtered_hbond_df['res_atom_pair'] = filtered_hbond_df.apply(
        lambda row: tuple(sorted([(row['res_1_name'], row['atom_1']), (row['res_2_name'], row['atom_2'])])),
        axis=1
    )

    # next, group by (res_1_name, res_2_name) as well as by atoms involved in the interaction
    # grouped_hbond_df = filtered_hbond_df.groupby(["res_1_name", "res_2_name", "atom_1", "atom_2"])
    grouped_hbond_df = filtered_hbond_df.groupby(['res_atom_pair'])

    # Finally, for each group, make heatmaps of (distance,angle)
    """    for group in grouped_hbond_df:
        group_name = group[0]
        type_1 = str(group_name[0])
        type_2 = str(group_name[1])
        atom_1 = str(group_name[2])
        atom_2 = str(group_name[3])

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        hbonds = group[1]
        hbonds_subset = hbonds[['distance', 'angle']]
        hbonds_subset = hbonds_subset.reset_index(drop=True)

        if (
                len(hbonds_subset) >= 100):  # & (len(hbonds_subset) <= 400): this limit existed before size limit was removed
            # Set global font size
            plt.rc('font', size=14)  # Adjust the font size as needed

            distance_bins = [i / 10 for i in range(20, 41)]  # Bins from 0 to 4 in increments of 0.1
            angle_bins = [i for i in range(0, 181, 10)]  # Bins from 0 to 180 in increments of 10

            hbonds_subset['distance_bin'] = pd.cut(hbonds_subset['distance'], bins=distance_bins)
            hbonds_subset['angle_bin'] = pd.cut(hbonds_subset['angle'], bins=angle_bins)

            heatmap_data = hbonds_subset.groupby(['angle_bin', 'distance_bin']).size().unstack(fill_value=0)

            plt.figure(figsize=(10, 10))
            sns.heatmap(heatmap_data, cmap='gray_r', xticklabels=1, yticklabels=range(0, 181, 10), square=True)

            plt.xticks(np.arange(len(distance_bins)) + 0.5, [f'{bin_val:.1f}' for bin_val in distance_bins], rotation=0)
            plt.yticks(np.arange(len(angle_bins)) + 0.5, angle_bins, rotation=0)

            plt.xlabel("Distance (angstroms)")
            plt.ylabel("Angle (degrees)")
            map_name = type_1 + "-" + type_2 + " " + atom_1 + "-" + atom_2
            plt.title(map_name + " H-bond heatmap")

            if len(type_1) == 1 and len(type_2) == 1:
                map_dir = "heatmaps/RNA-RNA"
            else:
                map_dir = "heatmaps/RNA-PROT"

            __safe_mkdir(map_dir)

            map_dir = map_dir + "/" + map_name
            plt.savefig(f"{map_dir}.png", dpi=250)
            plt.close()

            heatmap_csv_path = "heatmap_data"
            __safe_mkdir(heatmap_csv_path)

            heat_data_csv_path = heatmap_csv_path + "/" + map_name + ".csv"
            hbonds.to_csv(heat_data_csv_path, index=False)

            heatmap_res_names.append(map_name)
            heatmap_atom_names.append(len(hbonds_subset))

            # Insert the code for the 2D histogram here
            plt.figure(figsize=(10, 8))
            plt.hist2d(hbonds_subset['distance'], hbonds_subset['angle'], bins=[distance_bins, angle_bins],
                       cmap='gray_r')
            plt.xlabel("Distance (angstroms)")
            plt.ylabel("Angle (degrees)")
            plt.colorbar(label='Frequency')
            map_name = type_1 + "-" + type_2 + " " + atom_1 + "-" + atom_2
            plt.title(map_name + " H-bond heatmap")

            if len(type_1) == 1 and len(type_2) == 1:
                map_dir = "heatmaps/RNA-RNA"
            else:
                map_dir = "heatmaps/RNA-PROT"

            __safe_mkdir(map_dir)
            map_dir = map_dir + "/" + map_name
            # Save the 2D histogram as a PNG file
            plt.savefig(f"{map_dir}.png", dpi=250)
            # Sometimes the terminal might kill the process
            # if that happens lower the DPI setting above

            plt.close()  # Close the plot to prevent overlapping plots

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
    """

    # Process each group to make heatmaps; code above is optimized and this is the OG version
    # OG version makes more sense after cleaning it up
    for group_name, hbonds in grouped_hbond_df:
        # type_1, type_2, atom_1, atom_2 = map(str, group_name)
        (res_1, atom_1), (res_2, atom_2) = group_name
        type_1, type_2 = res_1, res_2

        print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
        hbonds_subset = hbonds[['distance', 'angle']].reset_index(drop=True)

        if len(hbonds_subset) >= 100:  # if there are more than 100 data points
            # Set global font size (was set up there)
            # plt.rc('font', size=18)

            distance_bins = np.arange(2.0, 4.1, 0.1)  # Bins from 2 to 4 in increments of 0.1
            angle_bins = np.arange(0, 181, 10)  # Bins from 0 to 180 in increments of 10

            hbonds_subset['distance_bin'] = pd.cut(hbonds_subset['distance'], bins=distance_bins)
            hbonds_subset['angle_bin'] = pd.cut(hbonds_subset['angle'], bins=angle_bins)

            heatmap_data = hbonds_subset.groupby(['angle_bin', 'distance_bin']).size().unstack(fill_value=0)

            plt.figure(figsize=(6, 6))
            sns.heatmap(heatmap_data, cmap='gray_r', xticklabels=1, yticklabels=range(0, 181, 10), square=True)

            plt.xticks(np.arange(len(distance_bins)) + 0.5, [f'{bin_val:.1f}' for bin_val in distance_bins], rotation=0)
            plt.yticks(np.arange(len(angle_bins)) + 0.5, angle_bins, rotation=0)

            plt.xlabel("Distance (Å)")
            plt.ylabel("Angle (°)")
            map_name = f"{type_1}-{type_2} {atom_1}-{atom_2}"
            # plt.title(f"{map_name} H-bond heatmap")

            map_dir = "heatmaps/RNA-RNA" if len(type_1) == 1 and len(type_2) == 1 else "heatmaps/RNA-PROT"
            __safe_mkdir(map_dir)

            map_path = f"{map_dir}/{map_name}.png"
            plt.savefig(map_path, dpi=250)
            plt.close()

            heatmap_csv_path = "heatmap_data"
            __safe_mkdir(heatmap_csv_path)

            heat_data_csv_path = f"{heatmap_csv_path}/{map_name}.csv"
            hbonds.to_csv(heat_data_csv_path, index=False)

            # 2D histogram
            plt.figure(figsize=(6, 4.8))
            plt.hist2d(hbonds_subset['distance'], hbonds_subset['angle'], bins=[distance_bins, angle_bins],
                       cmap='gray_r')
            plt.xlabel("Distance (Å)")
            plt.ylabel("Angle (°)")
            plt.colorbar(label='Count')
            plt.title(f"{map_name} H-bond heatmap")

            plt.savefig(map_path, dpi=250)
            plt.close()

        else:
            print(f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient data points.")

    # after collecting data make the final histogram of all the data in heatmaps
    # first compile the list into a df
    # gonna comment this out because this is for debug purposes
    # histo_df = pd.DataFrame({"heatmap": heatmap_res_names, "count": heatmap_atom_names})

    # plot histogram
    # plt.hist(histo_df.iloc[:, 1], bins=400)  # Adjust the number of bins as needed

    # set labels
    # plt.xlabel('# of datapoints inside a heatmap')
    # plt.ylabel('# of heatmaps with X datapoints')
    # plt.title('1d_histogram')

    # set y-axis limit
    # plt.ylim(0, max(df.iloc[:, 1]) * 0.9)

    # plt.savefig('1d_histo.png')
    # plt.close()


# optimized code
# added snap data for RNPs
# revised figures
# waiting on CSV data to fix heatmap code crashing issue
# need to start thinking about how to compile/distribute

# calculate some final statistics (i.e. Figure 2)
def __final_statistics():
    print("Plotting...")
    # graphs
    motif_directory = "/Users/jyesselm/PycharmProjects/rna_motif_library/rna_motif_library/motifs"

    # Create a dictionary to store counts for each folder
    folder_counts = {"TWOWAY": 0, "NWAY": 0, "HAIRPIN": 0, "HELIX": 0, "SSTRAND": 0}  # Initialize counts

    # for folder in directory, count numbers:
    # try:
    # Iterate over all items in the specified directory
    for item_name in os.listdir(motif_directory):
        item_path = os.path.join(motif_directory, item_name)

        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Check if the folder name is "2ways"
            if item_name == "2ways":
                # If folder name is "2ways", register the count as TWOWAY
                folder_counts["TWOWAY"] += file_count
            elif "ways" in item_name:
                # If folder name contains "ways" but is not "2ways", register the count as NWAY
                folder_counts["NWAY"] += file_count
            elif item_name == "hairpins":
                # If folder name is "hairpins", register the count as HAIRPIN
                folder_counts["HAIRPIN"] += file_count
            elif item_name == "helices":
                # If folder name is "helices", register the count as HELIX
                folder_counts["HELIX"] += file_count
            elif item_name == "sstrand":
                # If folder name is "sstrand", register count as SSTRAND
                folder_counts["SSTRAND"] += file_count
            else:
                # If the folder name doesn't match any condition, use it as is
                folder_counts[item_name] = file_count

    """# make a bar graph of all types of motifs
    folder_names = list(folder_counts.keys())
    file_counts = list(folder_counts.values())
    # Sort the folder names and file counts alphabetically
    folder_names_sorted, file_counts_sorted = zip(*sorted(zip(folder_names, file_counts)))
    # Set consistent parameters
    plt.rcParams.update({'font.size': 18})  # Set overall text size
    plt.figure(figsize=(8, 8))
    plt.bar(folder_names_sorted, file_counts_sorted, edgecolor='black', width=1)
    plt.xlabel('Motif Type')
    plt.ylabel('Count')
    plt.title('')  # Presence of N-way Junctions
    plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    # Save the graph as a PNG file
    plt.savefig('figure_2_bar_graph_motif_counts.png', dpi=600)
    # Don't display the plot
    plt.close()"""

    # Convert the dictionary to a DataFrame
    data = pd.DataFrame(list(folder_counts.items()), columns=['Motif Type', 'Count'])
    # Sort the DataFrame by 'Motif Type'
    data = data.sort_values('Motif Type')
    # Set the Seaborn theme
    sns.set_theme(style="white")  # palette='deep', color_codes=True)
    # Create the bar plot
    plt.figure(figsize=(6, 6), facecolor='white')
    plt.rcParams.update({'font.size': 20})  # Set overall text size
    sns.barplot(data=data, x='Motif Type', y='Count', color=sns.color_palette()[0], edgecolor='black')
    # Set labels and title
    plt.xlabel('Motif Type')
    plt.ylabel('Count')
    plt.title('')
    # Adjust layout for better fit
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig('figure_2_bar_graph_motif_counts.png', dpi=600)
    # Close the plot to avoid display
    plt.close()

    # of the hairpins, how long are they (histogram)
    hairpin_directory = motif_directory + "/hairpins"
    hairpin_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(hairpin_directory):
        item_path = os.path.join(hairpin_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            hairpin_counts[item_name] = file_count
    # Convert hairpin folder names to integers and sort them
    sorted_hairpin_counts = dict(sorted(hairpin_counts.items(), key=lambda item: int(item[0])))
    # Extract sorted keys and values
    hairpin_folder_names_sorted = list(sorted_hairpin_counts.keys())
    hairpin_file_counts_sorted = list(sorted_hairpin_counts.values())
    # Convert hairpin folder names to integers
    hairpin_bins = sorted([int(name) for name in hairpin_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(hairpin_bins), max(hairpin_bins) + 1)
    sns.set_theme(style='white')
    plt.figure(figsize=(6, 6))
    plt.hist(hairpin_bins, bins=np.arange(min(hairpin_bins) - 0.5, max(hairpin_bins) + 1.5, 1),
             weights=hairpin_file_counts_sorted, color=sns.color_palette()[0], edgecolor='black', align='mid',
             width=0.8)
    plt.xlabel('Hairpin Length')
    plt.ylabel('Count')
    plt.title('')  # Hairpins with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(np.arange(min(hairpin_bins), max(hairpin_bins) + 1, 5))  # Display ticks every 5 integers
    # plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig('figure_2_hairpin_counts_bar_graph.png', dpi=600)
    # Don't display the plot
    plt.close()

    # of the helices, how long are they (bar graph)
    helix_directory = motif_directory + "/helices"

    helix_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(helix_directory):
        item_path = os.path.join(helix_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            helix_counts[item_name] = file_count
    # Convert helix folder names to integers and sort them
    sorted_helix_counts = dict(sorted(helix_counts.items(), key=lambda item: int(item[0])))
    # Extract sorted keys and values
    helix_folder_names_sorted = list(sorted_helix_counts.keys())
    helix_file_counts_sorted = list(sorted_helix_counts.values())
    # Convert helix folder names to integers
    helix_bins = sorted([int(name) for name in helix_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(helix_bins), max(helix_bins) + 1)
    plt.figure(figsize=(6, 6))
    plt.hist(helix_bins, bins=np.arange(min(helix_bins) - 0.5, max(helix_bins) + 1.5, 1),
             weights=helix_file_counts_sorted, color=sns.color_palette()[0], edgecolor='black', align='mid', width=0.8)
    plt.xlabel('Helix Length')
    plt.ylabel('Count')
    plt.title('')  # Helices with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(np.arange(min(helix_bins), max(helix_bins) + 1, 5))  # Display ticks every 5 integers
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig('figure_2_helix_counts_bar_graph.png', dpi=600)
    # Don't display the plot
    plt.close()

    # Of the single strands, how long are they (bar graph)
    sstrand_directory = motif_directory + "/sstrand"

    sstrand_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(sstrand_directory):
        item_path = os.path.join(sstrand_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            sstrand_counts[item_name] = file_count
    # Convert helix folder names to integers and sort them
    sorted_sstrand_counts = dict(sorted(sstrand_counts.items(), key=lambda item: int(item[0])))
    # Extract sorted keys and values
    sstrand_folder_names_sorted = list(sorted_sstrand_counts.keys())
    sstrand_file_counts_sorted = list(sorted_sstrand_counts.values())
    # Convert helix folder names to integers
    sstrand_bins = sorted([int(name) for name in sstrand_folder_names_sorted])
    # Calculate the positions for the tick marks (midpoints between bins)
    tick_positions = np.arange(min(sstrand_bins), max(sstrand_bins) + 1)
    plt.figure(figsize=(6, 6))
    plt.hist(sstrand_bins, bins=np.arange(min(sstrand_bins) - 0.5, max(sstrand_bins) + 1.5, 1),
             weights=sstrand_file_counts_sorted, color=sns.color_palette()[0], width=0.8, edgecolor='black',
             align='mid')
    plt.xlabel('Single Strand Length')
    plt.ylabel('Count')
    plt.title('')  # Helices with Given Length
    # Set custom tick positions and labels
    plt.xticks(tick_positions, tick_positions)
    plt.xticks(np.arange(min(sstrand_bins), max(sstrand_bins) + 1, 5))  # Display ticks every 5 integers
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the bar graph as a PNG file
    plt.savefig('figure_2_sstrand_counts_bar_graph.png', dpi=600)
    # Don't display the plot
    plt.close()

    # create a bar graph of how many tertiary contacts are present
    tert_motif_directory = "/Users/jyesselm/PycharmProjects/rna_motif_library/rna_motif_library/tertiary_contacts"

    # Create a dictionary to store counts for each folder
    tert_folder_counts = {}
    # Iterate over all items in the specified directory
    for item_name in os.listdir(tert_motif_directory):
        item_path = os.path.join(tert_motif_directory, item_name)
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Perform your action for each folder
            file_count = count_files_with_extension(item_path, ".cif")
            # Store the count in the dictionary
            tert_folder_counts[item_name] = file_count
    # make a bar graph of all types of motifs
    tert_folder_names = list(tert_folder_counts.keys())
    tert_file_counts = list(tert_folder_counts.values())
    # Sort the folder names and file counts alphabetically
    tert_folder_names_sorted, tert_file_counts_sorted = zip(*sorted(zip(tert_folder_names, tert_file_counts)))
    plt.figure(figsize=(6, 6))
    plt.barh(tert_folder_names_sorted, tert_file_counts_sorted, edgecolor='black', color=sns.color_palette()[0],
             height=0.8)
    plt.xlabel('Count')
    plt.ylabel('Tertiary Contact Type')
    plt.title('')  # tertiary contact types
    # Adjust x-axis ticks for a tight fit
    # plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    # Save the graph as a PNG file
    plt.savefig('figure_3_tertiary_motif_counts.png', dpi=600)
    # Don't display the plot
    plt.close()

    # create a bar graph for how many tert contacts of each residue component are present
    tert_contact_csv_directory = 'unique_tert_contacts.csv'

    # Load the CSV into a DataFrame and keep only the columns 'res_type_1' and 'res_type_2'
    df = pd.read_csv(tert_contact_csv_directory, usecols=['res_type_1', 'res_type_2'])
    df = df.replace({'res_type_1': 'aa', 'res_type_2': 'aa'}, 'base')

    tuples_list = [tuple(sorted(x)) for x in df.to_records(index=False)]

    # Count the occurrences of each tuple
    tuple_counts = Counter(tuples_list)

    # Convert the counts to a DataFrame for easier plotting
    tuple_counts_df = pd.DataFrame(tuple_counts.items(), columns=['Contact Type', 'Count'])

    # Sort the DataFrame by Count (optional)
    tuple_counts_df = tuple_counts_df.sort_values(by='Count')

    # Plot the bar graph
    plt.figure(figsize=(6, 6))
    plt.barh(tuple_counts_df['Contact Type'].astype(str), tuple_counts_df['Count'], edgecolor='black',
             height=0.8, color=sns.color_palette()[0])
    plt.xlabel('Count')
    plt.ylabel('Tertiary Contact Type')
    plt.title('')  # tertiary contact types
    plt.tight_layout()

    # Save the graph as a PNG file
    plt.savefig('figure_3_residue_contact_types.png', dpi=600)

    # Now make it into a graph


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


# counts all files with a specific extension (used in generating Fig 2)
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


# sort 'res_1' and 'res_2' columns within each row
# this function has a very specific use case
def sort_res(row):
    return pd.Series(np.sort(row.values))


def main():
    warnings.filterwarnings("ignore")  # blocks the ragged nested sequence warning
    # time tracking stuff, tracks how long a process takes
    current_time = datetime.datetime.now()
    start_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Download of a nonredundant set
    csv_path = settings.LIB_PATH + "/data/csvs/nrlist_3.320_3.5A.csv"

    # __download_cif_files(csv_path)
    print("!!!!! CIF FILES DOWNLOADED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Download finished on", time_string)

    # Processing with DSSR
    # __get_dssr_files()
    print("!!!!! DSSR PROCESSING FINISHED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("DSSR processing finished on", time_string)

    # Processing with SNAP
    # __get_snap_files()
    print("!!!!! SNAP PROCESSING FINISHED !!!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("SNAP processing finished on", time_string)

    # Extracting motifs
    __generate_motif_files()
    print("!!!!! MOTIF EXTRACTION FINISHED !!!!!")
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Motif extraction finished on", time_string)

    # Finding tertiary contacts
    __find_tertiary_contacts()
    print("!!!!! TERTIARY CONTACT PROCESSING FINISHED !!!!!")

    # Printing heatmaps/plotting
    print("Printing heatmaps of data...")
    __heatmap_creation()

    # More plotting of other general data
    print("Plotting data...")
    __final_statistics()
    print("!!!!! PLOTS COMPLETED !!!!!")

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job started on", start_time_string)
    print("Job finished on", time_string)


if __name__ == '__main__':
    main()
