import os

import pandas as pd
from pydssr.dssr import DSSROutput


# writes extracted residue data into the proper output PDB files
def write_res_coords_to_pdb(nts, pdb_model, pdb_path):
    res = []
    for nt in nts:
        r = DSSRRes(nt)
        new_nt = r.chain_id + "." + str(r.num)
        # convert the MMCIF to a dictionary, and the resulting dictionary to a Dataframe
        dict = pdb_model.df
        df = pd.DataFrame.from_dict(dict, orient='index')
        model_df = df.iloc[0, 0]
        # sets up nucleotide IDs
        nt_id = new_nt.split(".")  # strings

        # Find residue in the PDB model
        chain_res = model_df[
            model_df['auth_asym_id'].astype(str) == str(nt_id[0])]  # first it picks the chain
        res_subset = chain_res[
            chain_res['auth_seq_id'].astype(str) == str(nt_id[1])]  # then it find the atoms
        # keeps certain columns from the CIF; trims and keeps 12 columns for rendering purposes
        res_subset = res_subset[
            ['group_PDB', 'id', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id',
             'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'type_symbol']]
        res.append(res_subset)  # "res" is a list with all the needed dataframes inside it

    s = ""
    df_list = []  # List to store the DataFrames for each line (type = 'list')
    for r in res:
        # Data reprocessing stuff, this loop is moving it into a DF
        lines = r.to_string(index=False, header=False).split('\n')
        for line in lines:
            values = line.split()  # (type of 'values' = list)
            # keeps the value error from happening by ignoring incorrectly-imported PDB files
            try:
                if len(values) == 12:
                    df = pd.DataFrame([values], columns=[
                        'group_PDB', 'id', 'label_atom_id', 'label_comp_id',
                        'label_asym_id', 'label_seq_id', 'Cartn_x',
                        'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv',
                        'type_symbol'
                    ])
                    df_list.append(df)
            except ValueError:
                continue
        if df_list:  # i.e. if there are things inside df_list:
            # Concatenate all DFs into a single DF
            result_df = pd.concat(df_list, axis=0, ignore_index=True)
            # Removes duplicate atoms in the DF (by coordinates)
            result_df = __remove_duplicate_lines(df=result_df)
            # skips DFs ones w/ no RNA by checking for phosphorous
            # if result_df['type_symbol'].str.contains('P').any():
            # renumber IDs so there are no more duplicate IDs and passes them as ints
            result_df['id'] = range(1, len(result_df) + 1)
            # updates the sequence IDs so they are all unique
            result_df = __reassign_unique_sequence_ids(result_df, 'label_seq_id')
            # writes the dataframe to a CIF file
            __dataframe_to_cif(df=result_df, file_path=f"{pdb_path}.cif")


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


def __reassign_unique_sequence_ids(df, column_name):
    current_id = 0
    prev_value = None

    for i, value in enumerate(df[column_name]):
        if value != prev_value:
            current_id += 1
        df.at[i, column_name] = current_id
        prev_value = value

    return df


# removes duplicate lines in DF
def __remove_duplicate_lines(df):
    # Initialize variables
    unique_coords = set()
    indices_to_drop = []

    # Iterate over the DataFrame
    for i, row in df.iterrows():
        # Check if coordinates are already encountered
        coords = (row['Cartn_x'], row['Cartn_y'], row['Cartn_z'])
        if coords in unique_coords:
            indices_to_drop.extend(range(i, len(df)))
            break  # Exit the loop once duplicate coordinates are found

        # Add coordinates to the set
        unique_coords.add(coords)

    # Remove the duplicate lines and reset the index
    df_unique = df.drop(indices_to_drop).reset_index(drop=True)
    return df_unique


# takes data from a dataframe and writes it to a CIF
def __dataframe_to_cif(df, file_path):
    # Open the CIF file for writing
    with open(file_path, 'w') as f:
        # Write the CIF header section
        f.write('data_\n')
        f.write('loop_\n')
        f.write('_atom_site.group_PDB\n')
        f.write('_atom_site.id\n')
        f.write('_atom_site.label_atom_id\n')
        f.write('_atom_site.label_comp_id\n')
        f.write('_atom_site.label_asym_id\n')
        f.write('_atom_site.label_seq_id\n')
        f.write('_atom_site.Cartn_x\n')
        f.write('_atom_site.Cartn_y\n')
        f.write('_atom_site.Cartn_z\n')
        f.write('_atom_site.occupancy\n')
        f.write('_atom_site.B_iso_or_equiv\n')
        f.write('_atom_site.type_symbol\n')
        # Write the data from the DataFrame (formatting)
        for row in df.itertuples(index=False):
            f.write("{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}{:<10}{:<10}{:<6}\n".format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                    row[10], row[11]
            ))