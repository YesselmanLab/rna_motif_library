import os
import re
import shutil
import math

import pandas as pd
import numpy as np
from pydssr.dssr import DSSROutput
from update_library import PandasMmcifOverride


# make new directories if they don't exist
def make_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# delete specified directories if they exist
def safe_delete_dir(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception as e:
            print(f"Error deleting directory '{directory_path}': {e}")


"""# separates CIF and PDB files after all is said and done (deprecated)
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

    print(f".pdb files deleted from {directory}")"""


# takes data from a dataframe and writes it to a CIF
def dataframe_to_cif(df, file_path, motif_name):
    # Open the CIF file for writing
    with open(file_path, 'w') as f:
        # Write the CIF header section; len(row) = 21
        f.write('data_\n')
        f.write('_entry.id ' + motif_name + '\n')
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
            f.write(
                "{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}{:<10}{:<10}{:<6}{:<6}{:<6}{:<6}{:<6}{:<6}\n".format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                    row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18],
                    row[19], row[20]
                ))


"""# takes data from a dataframe and writes it to a PDB (deprecated)
def dataframe_to_pdb(df, file_path):
    with open(file_path, 'w') as f:
        for row in df.itertuples(index=False):
            f.write("{:<5}{:>6}  {:<3} {:>3}{:>2}  {:>2}     {:>7} {:>7} {:>7}   {:>3} {:>3}         {:>3}\n".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                row[10], row[11]))"""


# remove empty dataframes
def remove_empty_dataframes(dataframes_list):
    dataframes_list = [df for df in dataframes_list if not df.empty]
    return dataframes_list


# extracts residue IDs properly from strings (protein extractions)
def extract_longest_numeric_sequence(input_string):
    longest_sequence = ""
    current_sequence = ""
    for c in input_string:
        if c.isdigit() or (c == '-' and (not current_sequence or current_sequence[0] == '-')):
            current_sequence += c
            if len(current_sequence) >= len(longest_sequence):
                longest_sequence = current_sequence
        else:
            current_sequence = ""

    return longest_sequence


def extract_longest_letter_sequence(input_string):
    # Find all sequences of letters using regular expression
    letter_sequences = re.findall('[a-zA-Z]+', input_string)

    # If there are no letter sequences, return an empty string
    if not letter_sequences:
        return ""

    # Find the longest letter sequence
    longest_sequence = max(letter_sequences, key=len)

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
    # Calculate the Euclidean distance between two points represented by DataFrames.
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

    return distance


# Define a function to check if two residues are connected
def calc_residue_distances(res_1, res_2):
    residue1 = res_1[1]
    residue2 = res_2[1]

    # Convert 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns to numeric
    residue1[['Cartn_x', 'Cartn_y', 'Cartn_z']] = residue1[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(
        pd.to_numeric)
    residue2[['Cartn_x', 'Cartn_y', 'Cartn_z']] = residue2[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(
        pd.to_numeric)

    # Extract relevant atom data for both residues
    atom1 = residue1[residue1['auth_atom_id'].str.contains("O3'", regex=True)]
    # delete hydrogens in the selection if there are any
    atom1 = atom1[~atom1['auth_atom_id'].str.contains("H", regex=False)]

    atom2 = residue2[residue2['auth_atom_id'].isin(["P"])]

    # Calculate the Euclidean distance between the two atoms

    distance = np.linalg.norm(atom2[['Cartn_x', 'Cartn_y', 'Cartn_z']].values - atom1[
        ['Cartn_x', 'Cartn_y', 'Cartn_z']].values)

    return distance


# counts # of strands given a table of residues (this is the initial count)
def count_connections(master_res_df, motif_name, twoway_jct_csv):
    # debug, prints master DF to a CSV for closer inspection
    # master_res_df.to_csv("model.csv", index=False)

    # step 1: make a list of all known residues
    # there are several cases where the IDs don't represent the actual residues, so we have to account for each one

    # Extract unique values from pdbx_PDB_ins_code column
    unique_ins_code_values = master_res_df["pdbx_PDB_ins_code"]
    unique_model_num_values = master_res_df["pdbx_PDB_model_num"]

    # Convert each unique value to lists
    unique_ins_code_values_list = unique_ins_code_values.astype(str).tolist()
    unique_model_num_values_list = unique_model_num_values.astype(str).tolist()

    ins_code_set_list = sorted(set(unique_ins_code_values_list))
    model_num_set_list = sorted(set(unique_model_num_values_list))

    # lay out each case
    if len(ins_code_set_list) > 1:
        grouped_res_dfs = master_res_df.groupby(
            ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code'])
    elif len(model_num_set_list) > 1:
        # here we might need to filter the DFs to keep only 1 pdbx_PDB_model_num
        filtered_master_df = master_res_df[master_res_df['pdbx_PDB_model_num'] == "1"]
        grouped_res_dfs = filtered_master_df.groupby(
            ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_model_num'])
    else:
        grouped_res_dfs = master_res_df.groupby(
            ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code'])

    # puts the grouped residues in a list
    res_list = []
    for grouped_residue in grouped_res_dfs:
        res_list.append(grouped_residue)

    # step 2: find all possible dataframe combinations
    # this should include non-unique combinations
    combinations_of_residues = []

    # Nested loop to generate combinations
    for i in range(len(res_list)):
        for j in range(len(res_list)):
            combo = (res_list[i], res_list[j])
            combinations_of_residues.append(combo)
    # combinations_of_residues = list(itertools.combinations(res_list, 2))

    # step 3: calculate distances for each pair of dataframes and put them in a separate list
    distances_btwn_residues = []
    for pair_of_residues in combinations_of_residues:
        distance_btwn_residues = calc_residue_distances(pair_of_residues[0], pair_of_residues[1])
        distances_btwn_residues.append(distance_btwn_residues)

    # this step takes the longest

    # step 4: put the two lists together into a big dataframe
    combined_combo_distance_df = pd.DataFrame(
        {'Residues': combinations_of_residues, 'Distances': distances_btwn_residues})

    # step 5: delete the lines with distance > 4
    connected_residues_df = combined_combo_distance_df[
        combined_combo_distance_df["Distances"] < 2.7]
    connected_residues_df_final = connected_residues_df[connected_residues_df["Distances"] != 0]

    # step 6: extract the column with the combinations and put it back inside a list, take out the DFs
    list_of_residue_combos = connected_residues_df_final["Residues"].tolist()
    list_of_ids = []  # this is the list of connected residues to process

    for combo in list_of_residue_combos:
        res_1_data = combo[0]
        res_2_data = combo[1]

        res_1_id = res_1_data[0]
        res_2_id = res_2_data[0]

        small_list = [res_1_id, res_2_id]
        list_of_ids.append(small_list)

    # Step 7: take the list of pairs and extract chains from them
    chains = extract_continuous_chains(list_of_ids)

    refined_chains = connect_continuous_chains(chains)

    ultra_refined_chains = refine_continuous_chains(refined_chains)

    # final_refined_chains = combine_remaining_chains(ultra_refined_chains)

    # Step 8: count lens
    len_chains = len(ultra_refined_chains)

    # Step 9: print the contents of 2-way motifs into a CSV (nucleotide data, motif names)
    # also change the names to make them match TWOWAY motifs, this will cause problems if you don't!
    if len_chains == 2:

        # Concatenate all inner lists and keep only unique values
        result_0 = []
        for inner_list in ultra_refined_chains[0]:
            result_0.extend(inner_list)

        result_1 = []
        for inner_list in ultra_refined_chains[1]:
            result_1.extend(inner_list)

        # lists unique NTs inside each strand
        unique_nucleotides_0 = list(set(result_0))
        unique_nucleotides_1 = list(set(result_1))

        len_0 = len(unique_nucleotides_0)
        len_1 = len(unique_nucleotides_1)

        class_0 = len_0 - 2
        class_1 = len_1 - 2

        # change all NWAY names to TWOWAY
        motif_type_list = motif_name.split(".")
        motif_type = motif_type_list[0]

        if motif_type == "NWAY":
            new_motif_type = "TWOWAY"
            new_motif_class = str(class_0) + "-" + str(class_1)
            new_motif_name = new_motif_type + "." + motif_type_list[1] + "." + new_motif_class + "." + motif_type_list[
                2] + "." + motif_type_list[3]
            motif_name = new_motif_name

        # motif_name, motif_type (NWAY/TWOWAY), nucleotides_in_strand_1, nucleotides_in_strand_2
        twoway_jct_csv.write(
            motif_name + "," + motif_type + "," + str(len_0) + "," + str(len_1) + "," + str(class_0) + "," + str(
                class_1) + "\n")  # + number of nucleotides, which can be found by length of each element in ultra refined chains

    return len_chains, motif_name


### extract -> connect -> refine_continuous_chains are part of a 3-step process that counts RNA strands to determine the # of basepair ends
### a basepair end is just a basepair at the end of two strands
### number of strands = number of basepair ends

# Extracts continuous chains from a list
def extract_continuous_chains(pair_list):
    chains = []

    for pair in pair_list:
        matched_chain = None

        for chain in chains:
            if chain[-1][1] == pair[0] or chain[0][0] == pair[1]:
                matched_chain = chain
                break
            elif chain[-1][1] == pair[1] or chain[0][0] == pair[0]:
                matched_chain = chain
                break

        if matched_chain:
            matched_chain.append(pair)

        else:
            chains.append([pair])

    return chains


# Puts together some continuous chains (don't get rid of commented out yet)
def connect_continuous_chains(chains):
    connected_chains = []

    for chain in chains:
        connected = False
        for current_chain in connected_chains:
            # chain is appended to current_chain

            if current_chain[-1][1] == chain[0][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][0] == chain[-1][1]:
                current_chain.insert(0, chain)
                connected = True
                break
            elif current_chain[-1][0] == chain[-1][1]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][1] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][0] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[-1][0] == chain[0][1]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[-1][1] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break

        if not connected:
            connected_chains.append(chain)

    return connected_chains


# Finds all the remaining possible connections of chains to return a final list of all the present strands
def refine_continuous_chains(input_lists):
    merged = []

    def merge_lists(list1, list2):
        for sub_list1 in list1:
            for sub_list2 in list2:
                if any(item1 in sub_list2 for item1 in sub_list1):
                    list1.extend(sub_list2)
                    list2.clear()
                    return list1

    for i in range(len(input_lists)):
        current_list = input_lists[i]
        for j in range(i + 1, len(input_lists)):
            item = input_lists[j]
            if merge_lists(current_list, item):
                break

        if current_list:  # Check if current_list is not empty
            merged.append(current_list)

    return merged


# Take the remaining chains, and process them continuously in a loop until they have all been combined into their final states
def combine_remaining_chains():
    pass


# writes extracted residue data into the proper output PDB files
def write_res_coords_to_pdb(nts, interactions, pdb_model, pdb_path, motif_bond_list, csv_file, residue_csv_list,
                            twoway_csv):
    # directory setup for later
    dir = pdb_path.split("/")
    sub_dir = dir[3].split(".")
    motif_name = dir[3]
    # motif extraction
    nt_list = []
    # list of residues
    res = []
    # convert the MMCIF to a dictionary, and the resulting dictionary to a Dataframe
    model_df_first = pdb_model.df
    # df to CSV for debug
    model_df_first.to_csv("model_df.csv", index=False)
    # keep only needed DF columns so further functions don't error
    columns_to_keep = ['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                       'label_alt_id', 'label_comp_id', 'label_asym_id',
                       'label_entity_id', 'label_seq_id',
                       'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                       'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge',
                       'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                       'auth_atom_id', 'pdbx_PDB_model_num']

    model_df = model_df_first[columns_to_keep]

    model_df.to_csv("model_df.csv", index=False)

    # extracts identification data from nucleotide list
    for nt in nts:
        # r = DSSRRes(nt)
        # splits nucleotide names (chain_id, type + res_id)
        nt_spl = nt.split(".")
        # purify IDs
        chain_id = nt_spl[0]
        # if nt_spl[1] contains a /, split it
        # of that, the first one is the chain_id and te second the res_id
        residue_id = extract_longest_numeric_sequence(nt_spl[1])

        if "/" in nt_spl[1]:
            # print("run")
            sub_spl = nt_spl[1].split("/")
            residue_id = sub_spl[1]

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
            # constructs PDB DF (deprecated)
            """pdb_columns = ['group_PDB', 'id', 'label_atom_id', 'label_comp_id',
                                   'auth_asym_id', 'auth_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                                   'occupancy', 'B_iso_or_equiv', 'type_symbol']
            pdb_df = df[pdb_columns]
            pdb_df_list.append(pdb_df)"""
    if df_list:  # i.e. if there are things inside df_list:
        # Concatenate all DFs into a single DF
        result_df = pd.concat(df_list, axis=0, ignore_index=True)

        # this is for NWAY/2WAY jcts
        if ((sub_dir[0] == "NWAY") or (sub_dir[0] == "TWOWAY")):
            basepair_ends, motif_name = count_connections(result_df, motif_name=motif_name,
                                              twoway_jct_csv=twoway_csv)  # you need a master DF of residues here
            # Write # of BP ends to the motif name (labeling of n-way junction)
            if not (basepair_ends == 1):
                new_path = dir[0] + "/" + str(basepair_ends) + "ways" + "/" + dir[2] + "/" + sub_dir[
                    2] + "/" + sub_dir[3]
                name_path = new_path + "/" + motif_name
                # writing the file to its place
                make_dir(new_path)
            else:
                # if only 1 basepair end it should be reclassified as a hairpin
                sub_dir[0] = "HAIRPIN"
        if (sub_dir[0] == "HAIRPIN"):
            # hairpins classified by the # of looped nucleotides at the top of the pin
            # two NTs in a hairpin are always canonical pairs so just: (len nts - 2)
            hairpin_bridge_length = len(nts) - 2
            # after classification into tri/tetra/etc
            hairpin_path = dir[0] + "/hairpins/" + str(hairpin_bridge_length)
            make_dir(hairpin_path)
            name_path = hairpin_path + "/" + motif_name
        if (sub_dir[0] == "HELIX"):
            # helices should be classified into folders by their # of basepairs
            # this should be very simple as the lengths are given in the motif names
            # also classify further by the sequence composition, this is also given in motif name
            helix_count = str(sub_dir[2])
            helix_comp = str(sub_dir[3])
            # after classification put em in the folders
            helix_path = dir[0] + "/helices/" + helix_count + "/" + helix_comp
            make_dir(helix_path)
            name_path = helix_path + "/" + motif_name

        # all results will use this, but the specific paths are changed above depending on what the motif is
        dataframe_to_cif(df=result_df, file_path=f"{name_path}.cif", motif_name=motif_name)

    # print list of residues in motif to CSV
    # print(nts)
    residue_csv_list.write(motif_name + ',' + ','.join(nts) + '\n')


    # if there are interactions, do this:
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

            if "/" in inter_spl[1]:
                sub_spl = inter_spl[1].split("/")
                inter_protein_id = sub_spl[1]

            # define protein ID
            new_inter = inter_chain_id + "." + inter_protein_id
            # add it to the list of proteins
            inter_list.append(new_inter)
            inter_id = new_inter.split(".")
            # Find proteins in the PDB model
            inter_chain = model_df[
                model_df['auth_asym_id'].astype(str) == inter_id[0]]  # first the chain
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
            # concatenate proteins with RNA
            result_inter_df = pd.concat(inter_df_list, axis=0, ignore_index=True)  # interactions
            # result_inter_df.to_csv("inter.csv", index=False)
            total_result_df = pd.concat([result_df, result_inter_df], ignore_index=True)

            # for JCTs
            if ((sub_dir[0] == "NWAY") or (sub_dir[0] == "TWOWAY")):
                # set a path for the interactions
                inter_new_path = "motif_interactions/" + str(basepair_ends) + "ways/" + dir[
                    2] + "/" + \
                                 sub_dir[2] + "/" + sub_dir[3]
                inter_name_path = inter_new_path + "/" + motif_name + ".inter"
                make_dir(inter_new_path)
            # for hairpins
            if (sub_dir[0] == "HAIRPIN"):
                inter_hairpin_path = "motif_interactions/hairpins"
                inter_name_path = inter_hairpin_path + "/" + motif_name + ".inter"
                make_dir(inter_hairpin_path)
            # for helices
            if (sub_dir[0] == "HELIX"):
                inter_helix_path = "motif_interactions/helices/" + str(helix_count) + "/" + str(
                    helix_comp)
                inter_name_path = inter_helix_path + "/" + motif_name + ".inter"
                make_dir(inter_helix_path)

            # writes interactions to CIF
            dataframe_to_cif(df=total_result_df, file_path=f"{inter_name_path}.cif", motif_name=motif_name)

        # extracting individual interactions:
        find_matching_interactions(interactions_filtered, motif_bond_list, model_df, motif_name,
                                   csv_file)


# extracts individual interactions out
# of the individual interactions, check if they are in the same motif
def find_matching_interactions(inter_from_PDB, list_of_inters, pdb_model_df, motif_name, csv_file):
    # csv_file is f.inter
    list_of_matching_interactions = []

    # find everything the interacting residues are interacting with
    for target_value in inter_from_PDB:

        for hbond_inter in list_of_inters:
            if target_value in hbond_inter:
                list_of_matching_interactions.append(hbond_inter)

    # make directories just in case
    make_dir("interactions/all")

    # then, of those interactions, find the appropriate residues and print them
    for interaction in list_of_matching_interactions:
        # interaction format: ('f.ARG54', '3.G167', 'NH2', 'O6', '3.977')

        # Obtains residues
        res_1 = interaction[0]
        res_2 = interaction[1]
        # ^ these are the residues you want to extract and check

        # splitting chain and res data
        res_1_chain_id, res_1_res_data = res_1.split(".")
        res_2_chain_id, res_2_res_data = res_2.split(".")

        # if there are slashes present:
        if "/" in res_1:
            res_1_split = res_1.split(".")
            res_1_inside = res_1_split[1]
            res_1_res_data_spl = res_1_inside.split("/")
            res_1_res_data = res_1_res_data_spl[1]

        if "/" in res_2:
            res_2_split = res_2.split(".")
            res_2_inside = res_2_split[1]
            res_2_res_data_spl = res_2_inside.split("/")
            res_2_res_data = res_2_res_data_spl[1]

        # purify IDs
        res_1_res_id = extract_longest_numeric_sequence(res_1_res_data)
        res_2_res_id = extract_longest_numeric_sequence(res_2_res_data)

        # first find the chains for res_1 and res_2
        res_1_inter_chain = pdb_model_df[pdb_model_df["auth_asym_id"].astype(str) == str(res_1_chain_id)]
        res_2_inter_chain = pdb_model_df[pdb_model_df["auth_asym_id"].astype(str) == str(res_2_chain_id)]

        # then find the residues for res_1 and res_2
        res_1_inter_res = res_1_inter_chain[
            res_1_inter_chain['auth_seq_id'].astype(str) == str(res_1_res_id)]
        res_2_inter_res = res_2_inter_chain[
            res_2_inter_chain['auth_seq_id'].astype(str) == str(res_2_res_id)]

        # and concatenate them
        res_1_res_2_result_df = pd.concat([res_1_inter_res, res_2_inter_res], axis=0,
                                          ignore_index=True)

        # write interactions to CSV
        # first determine what type res_1 and res_2 are
        res_1_type_list = res_1_inter_res["auth_comp_id"].unique().tolist()
        res_2_type_list = res_2_inter_res["auth_comp_id"].unique().tolist()

        res_1_type = res_1_type_list[0]
        res_2_type = res_2_type_list[0]

        # exclude all canonical pairs
        # atom extraction
        atom_1 = interaction[2]
        atom_2 = interaction[3]

        distance_ext = interaction[4]

        # calculate the angle in the interaction
        # find which residue contains the oxygen atom

        if "O" in interaction[2]:  # case for O-O/O-N interactions

            # interaction[0] is the oxygen residue
            oxygen_residue = res_1_inter_res
            second_residue = res_2_inter_res

            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'] == str(interaction[2])]
            second_atom = second_residue[second_residue['auth_atom_id'] == str(interaction[3])]

            if oxygen_atom.empty:
                if "O2" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O2")]

                if "OP1" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O1P")]
                if "OP2" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O2P")]
                if "OP3" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O3P")]
                if "O1P" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP1")]
                if "O2P" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP2")]
                if "O3P" in str(interaction[2]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP3")]

                if "." in str(interaction[2]):
                    split_interaction = interaction[2].split(".")
                    oxygen_atom = oxygen_residue[
                        oxygen_residue['auth_atom_id'].str.contains(split_interaction[0])]

                    if oxygen_atom.empty:
                        if "OP1" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O1P")]
                        if "OP2" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O2P")]
                        if "OP3" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O3P")]
                        if "O1P" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP1")]
                        if "O2P" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP2")]
                        if "O3P" in str(interaction[2]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("OP3")]

            if second_atom.empty:
                if "OP1" in str(interaction[3]):
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains("O1P")]
                if "OP2" in str(interaction[3]):
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains("O2P")]
                if "OP3" in str(interaction[3]):
                    second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O3P")]
                if "O1P" in str(interaction[3]):
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains("OP1")]
                if "O2P" in str(interaction[3]):
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains("OP2")]
                if "O3P" in str(interaction[3]):
                    second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP3")]

                if "." in str(interaction[3]):
                    split_interaction = interaction[3].split(".")
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains(split_interaction[0])]

                    if second_atom.empty:
                        if "OP1" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O1P")]
                        if "OP2" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O2P")]
                        if "OP3" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O3P")]
                        if "O1P" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP1")]
                        if "O2P" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP2")]
                        if "O3P" in str(interaction[3]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP3")]


        elif "O" in interaction[3]:  # N-O interactions

            oxygen_residue = res_2_inter_res
            second_residue = res_1_inter_res

            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'] == str(interaction[3])]
            second_atom = second_residue[second_residue['auth_atom_id'] == str(interaction[2])]

            if oxygen_atom.empty:
                if "OP1" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O1P")]
                if "OP2" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O2P")]
                if "OP3" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O3P")]
                if "O1P" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP1")]
                if "O2P" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP2")]
                if "O3P" in str(interaction[3]):
                    oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP3")]

                # split the string by the point
                if "." in str(interaction[3]):
                    split_interaction = interaction[3].split(".")
                    oxygen_atom = oxygen_residue[
                        oxygen_residue['auth_atom_id'].str.contains(split_interaction[0])]

                    if oxygen_atom.empty:
                        if "OP1" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O1P")]
                        if "OP2" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O2P")]
                        if "OP3" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str.contains("O3P")]
                        if "O1P" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP1")]
                        if "O2P" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP2")]
                        if "O3P" in str(interaction[3]):
                            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'].str_contains("OP3")]

            if second_atom.empty:
                if "." in str(interaction[2]):
                    split_interaction_2 = interaction[2].split(".")
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains(split_interaction_2[0])]

                    if second_atom.empty:
                        if "OP1" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O1P")]
                        if "OP2" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O2P")]
                        if "OP3" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("O3P")]
                        if "O1P" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP1")]
                        if "O2P" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP2")]
                        if "O3P" in str(interaction[2]):
                            second_atom = second_residue[second_residue['auth_atom_id'].str.contains("OP3")]


        else:  # case for N-N interactions

            # interaction[1] is the oxygen residue
            oxygen_residue = res_2_inter_res
            second_residue = res_1_inter_res

            oxygen_atom = oxygen_residue[oxygen_residue['auth_atom_id'] == str(interaction[3])]
            second_atom = second_residue[second_residue['auth_atom_id'] == str(interaction[2])]

            if oxygen_atom.empty:
                # split the string by the point
                if "." in str(interaction[3]):
                    split_interaction = interaction[3].split(".")
                    oxygen_atom = oxygen_residue[
                        oxygen_residue['auth_atom_id'].str.contains(split_interaction[0])]

            if second_atom.empty:
                if "." in str(interaction[2]):
                    split_interaction_2 = interaction[2].split(".")
                    second_atom = second_residue[
                        second_residue['auth_atom_id'].str.contains(split_interaction_2[0])]

        carbon_atom = find_closest_atom(oxygen_atom, res_1_res_2_result_df)

        fourth_atom = find_closest_atom(second_atom, res_1_res_2_result_df)

        # calculate planar bond angle
        bond_angle_degrees, o_atom_data, n_atom_data, c_atom_data = calculate_bond_angle(
            oxygen_atom, second_atom, carbon_atom, fourth_atom)

        # setting the name
        # then set extension for name
        name_inter = motif_name + "." + res_1 + "." + res_2 + "." + res_1_type + "." + res_2_type

        # sort to avoid commutative directories
        alpha_sorted_types = sorted([res_1_type, res_2_type])

        if not ((alpha_sorted_types[0] == "A" and alpha_sorted_types[1] == "U") or
                (alpha_sorted_types[0] == "C" and alpha_sorted_types[1] == "G") or
                (len(alpha_sorted_types[0]) > 1 and len(alpha_sorted_types[1]) > 1)):
            # folder assignment
            folder_name = alpha_sorted_types[0] + "-" + alpha_sorted_types[1]
            ind_folder_path = "interactions/all/" + folder_name + "/"
            make_dir(ind_folder_path)
            # replace strings
            name_inter_2 = name_inter.replace("/", "-")

            # file path; ind_folder_path = the folder path
            ind_inter_path = ind_folder_path + name_inter_2

            # print(name_inter_2)
            # NWAY.7PKQ.7-14-1.GGUAAUAUU-GAUGGAAAGCCGAAGG-CAC.0.3.A119.3.U193

            # fill blanks
            res_1_res_2_result_df.fillna(0, inplace=True)

            dataframe_to_cif(res_1_res_2_result_df, file_path=f"{ind_inter_path}.cif", motif_name=name_inter)

            if len(res_1_type) == 1:
                nt_1 = "nt"
            else:
                nt_1 = "aa"
            if len(res_2_type) == 1:
                nt_2 = "nt"
            else:
                nt_2 = "aa"

            # f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")
            csv_file.write(
                motif_name + ',' + res_1 + ',' + res_2 + ',' + res_1_type + ',' + res_2_type + ',' + atom_1 + ',' + atom_2 + ',' + distance_ext + ',' + bond_angle_degrees + ',' + nt_1 + ',' + nt_2 + "\n")


# find the closest atom for 3rd point in angle
def find_closest_atom(atom_A, whole_interaction):
    # initialize variable
    min_distance_row = None
    min_distance = float("inf")

    # iterate through interaction to find right atoms
    for _, row in whole_interaction.iterrows():

        row_df = row.to_frame().T

        current_distance = calc_distance(atom_A, row_df)

        if 0 < current_distance < min_distance:
            min_distance = current_distance
            min_distance_row = row

    min_distance_row_df = min_distance_row.to_frame().T

    return min_distance_row_df


# calc the bond angle (need to return tuple with the atoms I used to calculate)
def calculate_bond_angle(center_atom, second_atom, carbon_atom, fourth_atom):
    # center_atom = oxygen atom; second_atom = usually the nitrogen atom; carbon_atom = 3rd atom
    # center is P, second A, carbon C, fourth F

    # point P - center atom (where we are calculating angle)
    x1, y1, z1 = center_atom["Cartn_x"].to_list()[0], center_atom["Cartn_y"].to_list()[0], \
        center_atom["Cartn_z"].to_list()[0]
    # point A - second atom in the H-bond
    x2, y2, z2 = second_atom["Cartn_x"].to_list()[0], second_atom["Cartn_y"].to_list()[0], \
        second_atom["Cartn_z"].to_list()[0]
    # point C - carbon atom connected to the center atom
    x3, y3, z3 = carbon_atom["Cartn_x"].to_list()[0], carbon_atom["Cartn_y"].to_list()[0], \
        carbon_atom["Cartn_z"].to_list()[0]
    # point F - fourth atom connected to second atom in H-bond
    x4, y4, z4 = fourth_atom["Cartn_x"].to_list()[0], fourth_atom["Cartn_y"].to_list()[0], \
        fourth_atom["Cartn_z"].to_list()[0]

    # tuples passed as points
    P = (x1, y1, z1)
    A = (x2, y2, z2)
    C = (x3, y3, z3)
    F = (x4, y4, z4)

    # passed to numpy and vectors are calculated
    vector_AP = np.array(A) - np.array(P)
    vector_PC = np.array(C) - np.array(P)
    vector_FA = np.array(F) - np.array(A)

    # get cross products for planes
    vector_n1 = np.cross(vector_AP, vector_PC)
    vector_n2 = np.cross(vector_AP, vector_FA)

    dot_product = np.dot(vector_n1, vector_n2)

    magnitude_n1 = np.linalg.norm(vector_n1)
    magnitude_n2 = np.linalg.norm(vector_n2)

    cos_theta = dot_product / (magnitude_n1 * magnitude_n2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = str(np.degrees(angle_rad))

    if np.degrees(angle_rad) > 180.0:
        exit(0)

    # need to label and return the atoms that are used in calculation

    center_atom_type = center_atom["auth_atom_id"].to_list()[0]
    carbon_atom_type = carbon_atom["auth_atom_id"].to_list()[0]

    second_atom_type = second_atom["auth_atom_id"].to_list()[0]

    center_atom_data = (center_atom_type, x1, y1, z1)
    second_atom_data = (second_atom_type, x2, y2, z2)
    carbon_atom_data = (carbon_atom_type, x3, y3, z3)
    # data for 4th atom?

    return angle_deg, center_atom_data, second_atom_data, carbon_atom_data


# calc distance with DFs
def calc_distance(atom_df1, atom_df2):
    try:
        x1 = atom_df1["Cartn_x"].tolist()[0]
        y1 = atom_df1["Cartn_y"].tolist()[0]
        z1 = atom_df1["Cartn_z"].tolist()[0]

        x2 = atom_df2["Cartn_x"].tolist()[0]
        y2 = atom_df2["Cartn_y"].tolist()[0]
        z2 = atom_df2["Cartn_z"].tolist()[0]

        distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))

    except IndexError:
        distance = 0

    return distance


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
    motif_hbonds, motif_interactions, hbonds_in_motif = __assign_hbonds_to_motifs(motifs, hbonds,
                                                                                  shared)
    motifs = __remove_duplicate_motifs(motifs)
    motifs = __remove_large_motifs(motifs)
    return motifs, motif_hbonds, motif_interactions, hbonds_in_motif


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


# assign interactions to motifs
def __assign_hbonds_to_motifs(motifs, hbonds, shared):
    # all the data about the hbonds in this
    hbonds_in_motif = []

    motif_hbonds = {}
    motif_interactions = {}
    start_dict = {
        'base:base': 0, 'base:sugar': 0, 'base:phos': 0,
        'sugar:base': 0, 'sugar:sugar': 0, 'sugar:phos': 0, 'phos:base': 0, 'phos:sugar': 0,
        'phos:phos': 0, 'base:aa': 0, 'sugar:aa': 0, 'phos:aa': 0
    }
    for hbond in hbonds:
        # This retrives the data from the JSON extracted by DSSR
        # res 1/res 2 are the data we need, they define the actual residues/proteins in the interaction
        atom1, res1 = hbond.atom1_id.split("@")
        atom2, res2 = hbond.atom2_id.split("@")

        distance_bonds = str(hbond.distance)

        # write residue pairs for exports
        hbond_res_pair = (res1, res2, atom1, atom2, distance_bonds)
        hbonds_in_motif.append(hbond_res_pair)

        # Specifies what kind of biomolecules they are
        rt1, rt2 = hbond.residue_pair.split(":")
        # not really needed ^ (for the process I want to build)

        m1, m2 = None, None

        # m1 and m2 are the actual motifs where
        for m in motifs:
            # if one of the hbond residues interact with a residue in the motif then copy the hbond residue in
            if res1 in m.nts_long:
                m1 = m

            if res2 in m.nts_long:
                m2 = m

        if m1 == m2:
            continue

        # this creates a name for the interaction (I think); check if the motifs are not empty
        if m1 is not None and m2 is not None:
            names = sorted([m1.name, m2.name])
            key = names[0] + "-" + names[1]
            if key in shared:
                continue
        # this specifies the class of interaction (aa:base, aa:aa, etc)
        hbond_classes = __assign_hbond_class(atom1, atom2, rt1, rt2)

        # this counts the # of hydrogen bonds in each category
        if m1 is not None:
            if m1.name not in motif_hbonds:
                motif_hbonds[m1.name] = dict(start_dict)
                motif_interactions[m1.name] = []

            # sets hydrogen bond class
            hbond_class = hbond_classes[0] + ":" + hbond_classes[1]

            # increments the start_dict
            motif_hbonds[m1.name][hbond_class] += 1
            motif_interactions[m1.name].append(res2)
        if m2 is not None:
            if m2.name not in motif_hbonds:
                motif_hbonds[m2.name] = dict(start_dict)
                motif_interactions[m2.name] = []

            # sets hydrogen bond class
            hbond_class = hbond_classes[1] + ":" + hbond_classes[0]
            # increments the start dict
            if hbond_classes[1] == 'aa':
                hbond_class = hbond_classes[0] + ":" + hbond_classes[1]
            motif_hbonds[m2.name][hbond_class] += 1
            motif_interactions[m2.name].append(res1)

    return motif_hbonds, motif_interactions, hbonds_in_motif


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
