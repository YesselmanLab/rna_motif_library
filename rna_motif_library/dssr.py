import os
import shutil
import math
import itertools

import pandas as pd
import numpy as np
from pydssr.dssr import DSSROutput

error_counter = 0
error_counter_2 = 0


# make new directories if they don't exist
def make_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# separates CIF and PDB files after all is said and done
def cif_pdb_sort(directory):
    # Create a copy of the directory with "_PDB" suffix
    directory_copy = directory + '_PDB'
    shutil.copytree(directory, directory_copy)

    # Iterate over the files in the copied directory
    for root, dirs, files in os.walk(directory_copy):
        for file in files:
            if file.endswith('.cif'):
                # Construct the file path
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)

    print(f".cif files deleted from {directory_copy}")

    # Iterate over the files in the original directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb'):
                # Construct the file path
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)

    print(f".pdb files deleted from {directory}")


# takes data from a dataframe and writes it to a CIF
def dataframe_to_cif(df, file_path):
    # Open the CIF file for writing
    with open(file_path, 'w') as f:
        # Write the CIF header section; len(row) = 21
        f.write('data_\n')
        f.write('loop_\n')
        f.write('_atom_site.group_PDB\n')  # 0
        f.write('_atom_site.id\n')  # 1
        f.write('_atom_site.type_symbol\n')  # 2
        f.write('_atom_site.label_atom_id\n')  # 3
        f.write('_atom_site.label_alt_id\n')  # 4
        f.write('_atom_site.label_comp_id\n')  # 5
        f.write('_atom_site.label_asym_id\n')  # 6
        f.write('_atom_site.label_entity_id\n')  # 7
        f.write('_atom_site.label_seq_id\n')  # 8
        f.write('_atom_site.pdbx_PDB_ins_code\n')  # 9
        f.write('_atom_site.Cartn_x\n')  # 10
        f.write('_atom_site.Cartn_y\n')  # 11
        f.write('_atom_site.Cartn_z\n')  # 12
        f.write('_atom_site.occupancy\n')  # 13
        f.write('_atom_site.B_iso_or_equiv\n')  # 14
        f.write('_atom_site.pdbx_formal_charge\n')  # 15
        f.write('_atom_site.auth_seq_id\n')  # 16
        f.write('_atom_site.auth_comp_id\n')  # 17
        f.write('_atom_site.auth_asym_id\n')  # 18
        f.write('_atom_site.auth_atom_id\n')  # 19
        f.write('_atom_site.pdbx_PDB_model_num\n')  # 20
        # Write the data from the DataFrame (formatting)
        for row in df.itertuples(index=False):
            f.write("{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}{:<10}{:<10}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}\n".format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                    row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18],
                    row[19], row[20]
            ))


# takes data from a dataframe and writes it to a PDB
def dataframe_to_pdb(df, file_path):
    with open(file_path, 'w') as f:
        for row in df.itertuples(index=False):
            f.write("{:<5}{:>6}  {:<3} {:>3}{:>2}  {:>2}     {:>7} {:>7} {:>7}   {:>3} {:>3}         {:>3}\n".format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                    row[10], row[11]))


# remove empty dataframes
def remove_empty_dataframes(dataframes_list):
    dataframes_list = [df for df in dataframes_list if not df.empty]
    return dataframes_list


# extracts residue IDs properly from strings
def extract_longest_numeric_sequence(input_string):
    longest_sequence = ""
    current_sequence = ""
    for c in input_string:
        if c.isdigit() or (c == '-' and (not current_sequence or current_sequence[0] == '-')):
            current_sequence += c
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
        else:
            current_sequence = ""
    return longest_sequence


# removes duplicate items in a list, meant for removing duplicate residues in a chain
def remove_duplicate_strings(original_list):
    unique_list = []
    for item in original_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


# groups residues into their own chains for sequence counting
def group_residues_by_chain(input_list):
    # Create a dictionary to hold grouped and sorted residue IDs by chain ID
    chain_residues = {}

    # Create a dictionary to hold chain IDs for the grouped residues
    chain_ids_for_residues = {}

    # Iterate through the input_list
    for item in input_list:
        chain_id, residue_id = item.split(".")
        if residue_id != "None":
            residue_id = int(residue_id)

            # Create a list for the current chain_id if not already present
            if chain_id not in chain_residues:
                chain_residues[chain_id] = []

            # Append the residue_id to the corresponding chain_id's list
            chain_residues[chain_id].append(residue_id)

            # Store the chain_id for this residue in the dictionary
            if residue_id not in chain_ids_for_residues:
                chain_ids_for_residues[residue_id] = []
            chain_ids_for_residues[residue_id].append(chain_id)

    # Sort each chain's residue IDs and store them in the list of lists
    sorted_chain_residues = []
    sorted_chain_ids = []

    # Sort the chain IDs based on the order they appeared in the input
    unique_chain_ids = list(chain_residues.keys())

    # Sort the chain IDs in the order of appearance
    sorted_unique_chain_ids = unique_chain_ids

    for chain_id in sorted_unique_chain_ids:
        sorted_residues = sorted(set(chain_residues[chain_id]))
        sorted_chain_residues.append(sorted_residues)
        sorted_chain_ids.append(chain_id)

    return sorted_chain_residues, sorted_chain_ids


# calculates distance between atoms
def euclidean_distance_dataframe(df1, df2):
    global error_counter
    """Calculate the Euclidean distance between two points represented by DataFrames."""
    if {'Cartn_x', 'Cartn_y', 'Cartn_z'} != set(df1.columns) or {'Cartn_x', 'Cartn_y',
                                                                 'Cartn_z'} != set(df2.columns):
        raise ValueError("DataFrames must have 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns")

    try:
        point1 = df1[['Cartn_x', 'Cartn_y', 'Cartn_z']].values[0]
        point2 = df2[['Cartn_x', 'Cartn_y', 'Cartn_z']].values[0]
        squared_distances = [(float(x) - float(y)) ** 2 for x, y in zip(point1, point2)]
        distance = math.sqrt(sum(squared_distances))
    except IndexError:
        distance = 10
        error_counter += 1

    return distance


# Define a function to check if two residues are connected
def are_residues_connected(residue1, residue2):
    # Convert 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns to numeric
    residue1[['Cartn_x', 'Cartn_y', 'Cartn_z']] = residue1[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(
        pd.to_numeric)
    residue2[['Cartn_x', 'Cartn_y', 'Cartn_z']] = residue2[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(
        pd.to_numeric)

    # Extract relevant atom data for both residues
    atom1 = residue1[residue1['auth_atom_id'].str.contains("O3'", regex=True)]
    atom2 = residue2[residue2['auth_atom_id'].isin(["P"])]

    # Calculate the Euclidean distance between the two atoms
    # try:
    distance = np.linalg.norm(atom1[['Cartn_x', 'Cartn_y', 'Cartn_z']].values - atom2[
        ['Cartn_x', 'Cartn_y', 'Cartn_z']].values)
    # except ValueError:
    #    error_counter_2 += 1
    #    distance = 0
    # print(distance)
    return distance < 4.0


# actually iterates through the list of residues and finds strands
def find_strands(residue_list):
    # Group residues by 'auth_asym_id', 'auth_seq_id', and 'pdbx_PDB_ins_code'
    grouped = residue_list.groupby(['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code'])

    # Create a list of unique combinations of residues
    residue_combinations = list(itertools.combinations(grouped.groups.keys(), 2))

    strands = [] # List to store strands

    for combo in residue_combinations:
        residue1, residue2 = combo
        group1 = grouped.get_group(residue1) # first residue
        group2 = grouped.get_group(residue2) # second residue

        # Check if the residues are connected
        if are_residues_connected(group1, group2):
            # Determine if they belong to an existing strand or create a new one
            added_to_existing_strand = False
            for strand in strands:
                if residue1 in strand:
                    strand.add(residue2)
                    added_to_existing_strand = True
                    break
                elif residue2 in strand:
                    strand.add(residue1)
                    added_to_existing_strand = True
                    break

            # If not added to an existing strand, create a new strand
            if not added_to_existing_strand:
                new_strand = {residue1, residue2}
                strands.append(new_strand)
    return strands

"""    # Initialize a list to store the strands
    strands = []
    strands_df = []

    # Iterate through each residue in the list
    for i in range(len(residues)):
        residue_tuple = residues[i]

        residue = residue_tuple[1]

        # ID extraction
        residue_chain_ids = residue["auth_asym_id"]
        residue_res_ids = residue["auth_seq_id"]
        residue_ins_codes = residue["pdbx_PDB_ins_code"]
        # DF to lists
        residue_chain_ids_list = residue_chain_ids.tolist()
        residue_res_ids_list = residue_res_ids.tolist()
        residue_ins_codes_list = residue_ins_codes.tolist()
        # lists to unique values
        res_chain_id = list(set(residue_chain_ids_list))[0]
        res_res_id = list(set(residue_res_ids_list))[0]
        res_ins_code = list(set(residue_ins_codes_list))[0]

        # unique residue id
        unique_residue_id = res_chain_id + "." + res_res_id + "." + res_ins_code

        # Check if the residue is already part of another strand
        in_existing_strand = False
        for strand in strands:
            if unique_residue_id in strand:
                in_existing_strand = True
                break

        if not in_existing_strand:
            # Start a new strand with the current residue
            strand = [unique_residue_id]
            strand_df = [residue]

            # Iterate through the remaining residues
            j = i + 1
            while j < len(residues):
                current_residue = residues[j][1]

                # ID extraction
                current_residue_chain_ids = current_residue["auth_asym_id"]
                current_residue_res_ids = current_residue["auth_seq_id"]
                current_residue_ins_codes = current_residue["pdbx_PDB_ins_code"]
                # DF to lists
                current_residue_chain_ids_list = current_residue_chain_ids.tolist()
                current_residue_res_ids_list = current_residue_res_ids.tolist()
                current_residue_ins_codes_list = current_residue_ins_codes.tolist()
                # lists to unique values
                cur_res_chain_id = list(set(current_residue_chain_ids_list))[0]
                cur_res_res_id = list(set(current_residue_res_ids_list))[0]
                cur_res_ins_code = list(set(current_residue_ins_codes_list))[0]
                # unique residue id
                cur_unique_residue_id = cur_res_chain_id + "." + cur_res_res_id + "." + cur_res_ins_code

                # Check if the current residue is connected to the last residue in the strand
                if are_residues_connected(residue, current_residue):
                    strand.append(cur_unique_residue_id)
                    strand_df.append(current_residue)
                else:
                    # If not connected, break the loop
                    break

                j += 1

            # Add the strand to the list of strands
            strands.append(strand)
            strands_df.append(strand_df)

    #return len(strands)
"""



# literally just counts and returns the # of strands
def count_strands(master_res_df):
    # Step 1: get a list of residues
    # Group master_res_df by the specified columns
    grouped = master_res_df.groupby(['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code'])

    # Initialize a list to store the individual groups
    res_list = []

    # Iterate through the groups and append each group to res_list
    for group_data in grouped:
        res_list.append(group_data)
    # Now we have a list of residues to work with

    # pass through function that analyzes the residues to find which ones are in a strand
    list_of_strands = find_strands(master_res_df)

    return len(list_of_strands)


# writes extracted residue data into the proper output PDB files
def write_res_coords_to_pdb(nts, interactions, pdb_model, pdb_path):
    # directory setup for later
    dir = pdb_path.split("/")
    sub_dir = dir[3].split(".")
    motif_name = dir[3]
    # motif extraction
    nt_list = []
    # list of residues
    res = []
    # convert the MMCIF to a dictionary, and the resulting dictionary to a Dataframe
    model_df = pdb_model.df
    # extracts identification data from nucleotide list
    for nt in nts:
        # r = DSSRRes(nt)
        # splits nucleotide names
        nt_spl = nt.split(".")
        # purify IDs
        chain_id = nt_spl[0]
        residue_id = extract_longest_numeric_sequence(nt_spl[1])
        # define nucleotide ID
        new_nt = chain_id + "." + residue_id
        # add it to the list of nucleotides being processed
        nt_list.append(new_nt)
    # sorts nucleotide list for further processing
    nucleotide_list_sorted, chain_list_sorted = group_residues_by_chain(
            nt_list)  # nt_list_sorted is a list of lists

    # this list is for strand-counting purposes
    list_of_chains = []

    # extraction of residues into dataframes
    for chain_number, residue_list in zip(chain_list_sorted, nucleotide_list_sorted):
        for residue in residue_list:
            # Find residue in the PDB model, first it picks the chain
            chain_res = model_df[model_df['auth_asym_id'].astype(str) == str(chain_number)]
            res_subset = chain_res[
                chain_res['auth_seq_id'].astype(str) == str(residue)]  # then it find the atoms
            res.append(res_subset)  # "res" is a list with all the residue DFs inside
        list_of_chains.append(res)

    df_list = []  # List to store the DataFrames for each line (type = 'list')

    res = remove_empty_dataframes(res)
    for r in res:
        # Data reprocessing stuff, this loop is moving it into a DF
        lines = r.to_string(index=False, header=False).split('\n')
        for line in lines:
            values = line.split()  # (type 'values' = list)
            df = pd.DataFrame([values],
                              columns=['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                                       'label_alt_id', 'label_comp_id', 'label_asym_id',
                                       'label_entity_id', 'label_seq_id',
                                       'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                                       'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge',
                                       'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                                       'auth_atom_id', 'pdbx_PDB_model_num'])
            df_list.append(df)
            # constructs PDB DF
            """pdb_columns = ['group_PDB', 'id', 'label_atom_id', 'label_comp_id',
                                   'auth_asym_id', 'auth_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                                   'occupancy', 'B_iso_or_equiv', 'type_symbol']
            pdb_df = df[pdb_columns]
            pdb_df_list.append(pdb_df)"""
    if df_list:  # i.e. if there are things inside df_list:
        # Concatenate all DFs into a single DF
        result_df = pd.concat(df_list, axis=0, ignore_index=True)

        basepair_ends = len(find_strands(result_df))  # you need an actual list of residues here
        print(basepair_ends)
        new_path = dir[0] + "/" + str(basepair_ends) + "ways" + "/" + dir[2] + "/" + sub_dir[
            2] + "/" + sub_dir[3]
        name_path = new_path + "/" + motif_name
        make_dir(new_path)
        dataframe_to_cif(df=result_df, file_path=f"{name_path}.cif")

    if interactions != None:
        # removes duplicate amino acids
        interactions_filtered = remove_duplicate_strings(interactions)
        # interaction processing
        inter_list = []
        inter_res = []
        for inter in interactions_filtered:
            inter_spl = inter.split(".")
            # purify IDs
            inter_chain_id = inter_spl[0]
            inter_protein_id = extract_longest_numeric_sequence(inter_spl[1])
            # define protein ID
            new_inter = inter_chain_id + "." + inter_protein_id
            # add it to the list of proteins
            inter_list.append(new_inter)
            inter_id = new_inter.split(".")
            # Find proteins in the PDB model
            inter_chain = model_df[model_df['auth_asym_id'].astype(str) == inter_id[0]]
            inter_res_subset = inter_chain[inter_chain['auth_seq_id'].astype(str) == str(
                    inter_id[1])]  # then it find the atoms
            inter_res.append(
                    inter_res_subset)  # "res" is a list with all the needed dataframes inside it
        inter_df_list = []  # List to store the DataFrames for each line (type = 'list')
        inter_res = remove_empty_dataframes(inter_res)

        for inter in inter_res:
            # Data reprocessing stuff, this loop is moving it into a DF
            lines = inter.to_string(index=False, header=False).split('\n')
            for line in lines:
                values = line.split()  # (type 'values' = list)
                inter_df = pd.DataFrame([values],
                                        columns=['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                                                 'label_alt_id', 'label_comp_id', 'label_asym_id',
                                                 'label_entity_id', 'label_seq_id',
                                                 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y',
                                                 'Cartn_z',
                                                 'occupancy', 'B_iso_or_equiv',
                                                 'pdbx_formal_charge',
                                                 'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                                                 'auth_atom_id', 'pdbx_PDB_model_num'])
                inter_df_list.append(inter_df)
        if df_list and inter_df_list:
            result_inter_df = pd.concat(inter_df_list, axis=0, ignore_index=True)
            total_result_df = pd.concat([result_df, result_inter_df], ignore_index=True)
            inter_new_path = "motif_interactions/" + str(basepair_ends) + "ways/" + dir[2] + "/" + \
                             sub_dir[2] + "/" + sub_dir[3]
            inter_name_path = inter_new_path + "/" + motif_name + ".inter"
            make_dir(inter_new_path)
            # writes interactions to CIF
            dataframe_to_cif(df=total_result_df, file_path=f"{inter_name_path}.cif")

        # if pdb_df_list:  # i.e. if there are things inside pdb_df_list
        # pdb_result_df = pd.concat(pdb_df_list, axis=0, ignore_index=True)
        # writes the dataframe to a PDB file
        # dataframe_to_pdb(df=pdb_result_df, file_path=f"{pdb_path}.pdb")


class DSSRRes(object):
    def __init__(self, s):
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        i_num = 0
        for i, c in enumerate(spl[1]):
            if c.isdigit():
                cur_num = spl[1][i:]
                cur_num = extract_longest_numeric_sequence(cur_num)
                i_num = i
                break
        self.num = None
        try:
            if cur_num is not None:
                self.num = int(cur_num)
        except ValueError:
            pass
        self.chain_id = spl[0]
        self.res_id = spl[1][0:i_num]


def get_motifs_from_structure(json_path):
    name = os.path.splitext(json_path.split("/")[-1])[0]
    d_out = DSSROutput(json_path=json_path)
    motifs = d_out.get_motifs()
    motifs = __merge_singlet_seperated(motifs)
    __name_motifs(motifs, name)
    shared = __find_motifs_that_share_basepair(motifs)
    hbonds = d_out.get_hbonds()
    motif_hbonds, motif_interactions = __assign_hbonds_to_motifs(motifs, hbonds, shared)
    motifs = __remove_duplicate_motifs(motifs)
    motifs = __remove_large_motifs(motifs)
    return motifs, motif_hbonds, motif_interactions


def __assign_atom_group(name):
    if name == 'OP1' or name == 'OP2' or name == 'P':
        return "phos"
    elif name.endswith('\''):
        return "sugar"
    else:
        return "base"


def __assign_hbond_class(atom1, atom2, rt1, rt2):
    classes = []
    for a, r in zip([atom1, atom2], [rt1, rt2]):
        if r == 'nt':
            classes.append(__assign_atom_group(a))
        else:
            classes.append('aa')
    return classes


def __assign_hbonds_to_motifs(motifs, hbonds, shared):
    motif_hbonds = {}
    motif_interactions = {}
    start_dict = {
        'base:base' : 0, 'base:sugar': 0, 'base:phos': 0,
        'sugar:base': 0, 'sugar:sugar': 0, 'sugar:phos': 0, 'phos:base': 0, 'phos:sugar': 0,
        'phos:phos' : 0, 'base:aa': 0, 'sugar:aa': 0, 'phos:aa': 0
    }
    for hbond in hbonds:
        atom1, res1 = hbond.atom1_id.split("@")
        atom2, res2 = hbond.atom2_id.split("@")
        rt1, rt2 = hbond.residue_pair.split(":")
        m1, m2 = None, None
        for m in motifs:
            if res1 in m.nts_long:
                m1 = m
            if res2 in m.nts_long:
                m2 = m
        if m1 == m2:
            continue
        if m1 is not None and m2 is not None:
            names = sorted([m1.name, m2.name])
            key = names[0] + "-" + names[1]
            if key in shared:
                continue
        hbond_classes = __assign_hbond_class(atom1, atom2, rt1, rt2)
        if m1 is not None:
            if m1.name not in motif_hbonds:
                motif_hbonds[m1.name] = dict(start_dict)
                motif_interactions[m1.name] = []

            hbond_class = hbond_classes[0] + ":" + hbond_classes[1]
            motif_hbonds[m1.name][hbond_class] += 1
            motif_interactions[m1.name].append(res2)
        if m2 is not None:
            if m2.name not in motif_hbonds:
                motif_hbonds[m2.name] = dict(start_dict)
                motif_interactions[m2.name] = []
            hbond_class = hbond_classes[1] + ":" + hbond_classes[0]
            if hbond_classes[1] == 'aa':
                hbond_class = hbond_classes[0] + ":" + hbond_classes[1]
            motif_hbonds[m2.name][hbond_class] += 1
            motif_interactions[m2.name].append(res1)
    return motif_hbonds, motif_interactions


def __remove_duplicate_motifs(motifs):
    duplicates = []
    for m1 in motifs:
        if m1 in duplicates:
            continue
        m1_nts = []
        for nt in m1.nts_long:
            m1_nts.append(nt.split(".")[1])
        for m2 in motifs:
            if m1 == m2:
                continue
            m2_nts = []
            for nt in m2.nts_long:
                m2_nts.append(nt.split(".")[1])
            if m1_nts == m2_nts:
                duplicates.append(m2)
    unique_motifs = []
    for m in motifs:
        if m in duplicates:
            continue
        unique_motifs.append(m)
    return unique_motifs


def __remove_large_motifs(motifs):
    new_motifs = []
    for m in motifs:
        if len(m.nts_long) > 35:
            continue
        new_motifs.append(m)
    return new_motifs


def __merge_singlet_seperated(motifs):
    junctions = []
    others = []
    for m in motifs:
        if m.mtype == 'STEM' or m.mtype == 'HAIRPIN' or m.mtype == 'SINGLE_STRAND':
            others.append(m)
        else:
            junctions.append(m)
    merged = []
    used = []
    for m1 in junctions:
        m1_nts = m1.nts_long
        if m1 in used:
            continue
        for m2 in junctions:
            if m1 == m2:
                continue
            included = 0
            for r in m2.nts_long:
                if r in m1_nts:
                    included += 1
            if included < 2:
                continue
            for nt in m2.nts_long:
                if nt not in m1.nts_long:
                    m1.nts_long.append(nt)
            used.append(m1)
            used.append(m2)
            merged.append(m2)
    new_motifs = others
    for m in junctions:
        if m in merged:
            continue
        new_motifs.append(m)
    return new_motifs


def __find_motifs_that_share_basepair(motifs):
    pairs = {}
    for m1 in motifs:
        m1_nts = m1.nts_long
        for m2 in motifs:
            if m1 == m2:
                continue
            included = 0
            for r in m2.nts_long:
                if r in m1_nts:
                    included += 1
            if included < 2:
                continue
            names = sorted([m1.name, m2.name])
            key = names[0] + "-" + names[1]
            pairs[key] = 1
    return pairs


def __get_strands(motif):
    nts = motif.nts_long
    strands = []
    strand = []
    for nt in nts:
        r = DSSRRes(nt)
        if len(strand) == 0:
            strand.append(r)
            continue
        if r.num is None:
            r.num = 0
        if strand[-1].num is None:
            strand[-1].num = 0
        diff = strand[-1].num - r.num
        if diff == -1:
            strand.append(r)
        else:
            strands.append(strand)
            strand = [r]
    strands.append(strand)
    return strands


def __name_junction(motif, pdb_name):
    nts = motif.nts_long
    strands = __get_strands(motif)
    strs = []
    lens = []
    for strand in strands:
        s = "".join([x.res_id for x in strand])
        strs.append(s)
        lens.append(len(s) - 2)
    if len(strs) == 2:
        name = "TWOWAY."
    else:
        name = "NWAY."
    name += pdb_name + "."
    name += "-".join([str(l) for l in lens]) + "."
    name += "-".join(strs)
    return name


def __name_motifs(motifs, name):
    for m in motifs:
        m.nts_long = sorted(m.nts_long, key=__sorted_res_int)
    motifs = sorted(motifs, key=__sort_res)
    count = {}
    for m in motifs:
        if m.mtype == 'JUNCTION' or m.mtype == 'BULGE' or m.mtype == 'ILOOP':
            m_name = __name_junction(m, name)
        else:
            mtype = m.mtype
            if mtype == 'STEM':
                mtype = 'HELIX'
            elif mtype == 'SINGLE_STRAND':
                mtype = 'SSTRAND'
            m_name = mtype + "." + name + "."
            strands = __get_strands(m)
            strs = []
            for strand in strands:
                s = "".join([x.res_id for x in strand])
                strs.append(s)
            if mtype == 'HELIX':
                if len(strs) != 2:
                    m.name = 'UNKNOWN'
                    continue
                m_name += str(len(strands[0])) + "."
                m_name += strs[0] + "-" + strs[1]
            elif mtype == 'HAIRPIN':
                m_name += str(len(strs[0]) - 2) + "."
                m_name += strs[0]
            else:
                m_name += str(len(strs[0])) + "."
                m_name += strs[0]
        if m_name not in count:
            count[m_name] = 0
        else:
            count[m_name] += 1
        m.name = m_name + "." + str(count[m_name])


def __sorted_res_int(item):
    spl = item.split(".")
    return (spl[0], spl[1][1:])


def __sort_res(item):
    spl = item.nts_long[0].split(".")
    return (spl[0], spl[1][1:])
