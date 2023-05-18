import os
import json

import pandas as pd
from pydssr.dssr import DSSROutput
from biopandas.mmcif.pandas_mmcif import PandasMmcif


def pretty_print_json_file(input_file_path, output_file_path):
    # Read the contents of the input file
    with open(input_file_path, 'r') as f:
        input_json = json.load(f)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Write the pretty-printed JSON to the output file
    with open(output_file_path, 'w') as f:
        json.dump(input_json, f, indent=4, sort_keys=True)


# this one isn't actually used yet lmao
def write_motif_coords_to_pdbs(motifs, pdb_file):
    # Load the PDB file into a DataFrame (actually mmcif)
    df = PandasMmcif().read_mmcif(pdb_file)
    count = 0
    for m in motifs:
        # Select the subset of the structure corresponding to the motif
        subset = extract_structure(df, residue_ids=m.nts_long)
        # Generate a PDB-formatted string for the subset
        pdb_string = structure_to_pdb_string(subset)
        # Write the PDB-formatted string to a file
        with open(f"{m.mtype}.{count}.pdb", "w") as f:
            f.write(pdb_string)
        count += 1


def write_res_coords_to_pdb(nts, pdb_model, pdb_path):
    res = []
    for nt in nts:
        r = DSSRRes(nt)
        new_nt = r.chain_id + "." + str(r.num)
        # convert the MMCIF to a dictionary, and the resulting dictionary to a DF
        dict = pdb_model.df
        df = pd.DataFrame.from_dict(dict, orient='index')
        model_df = df.iloc[0, 0]

        # model_df.to_csv('output.csv', index=False)
        # Find residue in the PDB model
        atom_res = model_df['label_seq_id'] == new_nt
        chain_res = model_df['label_asym_id'] == r.chain_id
        res_subset = model_df[atom_res | chain_res]
        # keeps certain columns from the PDB
        res_subset = res_subset[
            ['group_PDB', 'id', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id',
             'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'type_symbol']]
        # resets temp factors
        #res_subset['B_iso_or_equiv'] = 0
        # Renumber the 'id' column
        res_subset['id'] = range(1, len(res_subset) + 1)
        res.append(res_subset)
    s = ""
    for r in res:
        lines = r.to_string(index=False, header=False).split('\n')
        df_list = []  # List to store the DataFrames for each line
        for line in lines:
            values = line.split()
            df = pd.DataFrame([values],
                              columns=['group_PDB', 'id', 'label_atom_id', 'label_comp_id',
                                       'label_asym_id', 'label_seq_id', 'Cartn_x',
                                       'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv',
                                       'type_symbol'])
            df_list.append(df)
        # Concatenate all DataFrames into a single DataFrame
        result_df = pd.concat(df_list, ignore_index=True)
        new_lines = []  # initializes a list of lines
        for _, row in result_df.iterrows():
            # goes through every row in the dataframe to reformat
            line = atom_to_atom_line(row)
            new_lines.append(line)
        s += "\n".join(new_lines) + '\n'
    with open(f"{pdb_path}.pdb", "w") as f:
        f.write(s)


def atom_to_atom_line(row):
    group_id = str(row['group_PDB'])
    atom_number = str(row['id'])
    atom_number = f"{atom_number:>6}" if len(atom_number) < 6 else atom_number

    atom_name = str(row['label_atom_id'])
    atom_name = f" {atom_name:<4}" if len(atom_name) < 4 else atom_name

    residue_name = str(row['label_comp_id'])
    chain_id = str(row['label_asym_id'])
    residue_number = str(row['label_seq_id'])
    x_coord = f"{str(row['Cartn_x']):>8}"
    y_coord = f"{str(row['Cartn_y']):>8}"
    z_coord = f"{str(row['Cartn_z']):>8}"
    occupancy = f"{str(row['occupancy']):>6}"
    temp_factor = f"{str(row['B_iso_or_equiv']):>6}"
    element_symbol = str(row['type_symbol'])
    # formatting was a bitch please don't touch unless you know what you're doing
    line = f"{group_id} {atom_number} {atom_name}{residue_name} {chain_id}   {residue_number}{' ' * 3} {x_coord}{y_coord}{z_coord}{occupancy}{temp_factor}{' ' * 11}{element_symbol}"
    return line


def extract_structure(input_mmcif, chain_ids=None, residue_ids=None, residue_names=None,
                      atom_names=None, element_symbols=None):
    # Passes the mmcif to a dictionary
    dict = input_mmcif.df
    # Converts the resulting dictionary into a Dataframe for further processing
    df = pd.DataFrame.from_dict(dict, orient='index')
    inner_df = df.iloc[0, 0]
    # Filter by chain IDs
    if chain_ids is not None:
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        inner_df = inner_df[inner_df['label_asym_id'].isin(chain_ids)]
    # Filter by residue IDs
    if residue_ids is not None:
        if isinstance(residue_ids, str):
            residue_ids = [residue_ids]
        inner_df = inner_df[inner_df['label_seq_id'].isin(residue_ids)]
    # Filter by residue names
    if residue_names is not None:
        if isinstance(residue_names, str):
            residue_names = [residue_names]
        inner_df = inner_df[inner_df['label_comp_id'].isin(residue_names)]
    # Filter by atom names
    if atom_names is not None:
        if isinstance(atom_names, str):
            atom_names = [atom_names]
        inner_df = inner_df[inner_df['label_atom_id'].isin(atom_names)]
    # Filter by element symbols
    if element_symbols is not None:
        if isinstance(element_symbols, str):
            element_symbols = [element_symbols]
        inner_df = inner_df[inner_df['type_symbol'].isin(element_symbols)]
    return inner_df.copy().reset_index(drop=True)


def structure_to_pdb_string(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(atom_to_atom_line(row))
    return '\n'.join(lines)


class DSSRRes(object):
    def __init__(self, s):
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        i_num = 0
        for i, c in enumerate(spl[1]):
            if c.isdigit():
                cur_num = spl[1][i:]
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
        name = 'NWAY.'
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
