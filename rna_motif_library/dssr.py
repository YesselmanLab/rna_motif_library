import os
import shutil

import pandas as pd
from pydssr.dssr import DSSROutput


# make new directories
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


# dataframe to PDB
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
        if c.isdigit():
            current_sequence += c
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = ""
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence
    return longest_sequence


# groups residues into their own chains
def group_residues_by_chain(input_list):
    # Create a dictionary to hold grouped and sorted residue IDs by chain ID
    chain_residues = {}

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

    # Sort each chain's residue IDs and store them in the list of lists
    sorted_chain_residues = []
    for chain_id, residues in chain_residues.items():
        sorted_residues = sorted(set(residues))
        sorted_chain_residues.append(sorted_residues)

    return sorted_chain_residues


# finds # of consecutive IDs (i.e. individual strands) as a proxy for basepair ends
def find_sequences(input_list):
    # first sort through the list of nucleotides by chain ID
    sorted_nt_ids_by_chain = group_residues_by_chain(input_list)
    # count the # of consecutive number sequences
    num_sequences = len(sorted_nt_ids_by_chain)
    for inner_list in sorted_nt_ids_by_chain:
        nt_ids = sorted(set(inner_list))
        # sequence counter, counts # of consecutive number sequences
        for i in range(len(nt_ids)):
            current_id = nt_ids[i]
            if i + 1 < len(nt_ids):
                next_id = nt_ids[i + 1]
                difference = next_id - current_id
                if difference != 1:
                    num_sequences += 1
    return num_sequences


# writes extracted residue data into the proper output PDB files
def write_res_coords_to_pdb(nts, pdb_model, pdb_path):
    # directory setup for later
    dir = pdb_path.split("/")
    sub_dir = dir[3].split(".")
    # motif extraction
    nt_list = []
    res = []
    # convert the MMCIF to a dictionary, and the resulting dictionary to a Dataframe
    model_df = pdb_model.df
    model_df.to_csv("df.csv", index=False)

    for nt in nts:
        r = DSSRRes(nt)
        # splits nucleotide names
        nt_spl = nt.split(".")
        # purify IDs
        chain_id = nt_spl[0]
        residue_id = extract_longest_numeric_sequence(nt_spl[1])
        # define nucleotide ID
        new_nt = chain_id + "." + residue_id
        # add it to the list of nucleotides being processed
        nt_list.append(new_nt)
        # sets up nucleotide IDs for further processing
        nt_id = new_nt.split(".")  # strings
        # Find residue in the PDB model
        chain_res = model_df[model_df['auth_asym_id'].astype(str) == nt_id[0]]
        res_subset = chain_res[chain_res['auth_seq_id'].astype(str) == str(nt_id[1])]  # then it find the atoms
        res.append(res_subset)  # "res" is a list with all the needed dataframes inside it
    df_list = []  # List to store the DataFrames for each line (type = 'list')
    # pdb_df_list = []
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

        if dir[0] != "motif_interactions":
            # this sorts and filters IDs so they are consecutive, and finds the # of strands
            basepair_ends = find_sequences(nt_list)
            new_path = dir[0] + "/" + str(basepair_ends) + "ways" + "/" + dir[2] + "/" + sub_dir[2] + "/" + \
                           sub_dir[
                               3]

        else:
            new_path = pdb_path
        make_dir(new_path)
        # writes the dataframe to a CIF file
        dataframe_to_cif(df=result_df, file_path=f"{new_path}.cif")
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
