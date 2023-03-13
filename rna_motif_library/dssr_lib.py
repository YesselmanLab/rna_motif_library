import os
from pydssr.dssr import DSSROutput
from biopandas.pdb.pandas_pdb import PandasPdb


def write_motif_coords_to_pdbs(motifs, pdb_file):
    # Load the PDB file into a DataFrame
    df = PandasPdb().read_pdb(pdb_file)
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


def write_res_coords_to_pdb(chain_ids, pdb_df, pdb_path):
    # Extract the selected atoms based on chain IDs
    pdb_subset = extract_structure(pdb_df, chain_ids=chain_ids)
    # Group the atoms by residue and write each residue to a PDB-formatted string
    pdb_strings = []
    for name, group in pdb_subset.groupby(['residue_name', 'residue_number', 'chain_id']):
        pdb_strings.append(structure_to_pdb_string(group))
    # Concatenate the PDB-formatted strings into a single string
    pdb_string = '\n'.join(pdb_strings)
    # Write the PDB-formatted string to a file
    with open(f"{pdb_path}.pdb", "w") as f:
        f.write(pdb_string)


def extract_structure(df, chain_ids=None, residue_ids=None, residue_names=None,
                      atom_names=None, element_symbols=None):
    # Filter by chain IDs
    if chain_ids is not None:
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        df = df[df['chain_id'].isin(chain_ids)]
    # Filter by residue IDs
    if residue_ids is not None:
        if isinstance(residue_ids, str):
            residue_ids = [residue_ids]
        df = df[df['residue_id'].isin(residue_ids)]
    # Filter by residue names
    if residue_names is not None:
        if isinstance(residue_names, str):
            residue_names = [residue_names]
        df = df[df['residue_name'].isin(residue_names)]
    # Filter by atom names
    if atom_names is not None:
        if isinstance(atom_names, str):
            atom_names = [atom_names]
        df = df[df['atom_name'].isin(atom_names)]
    # Filter by element symbols
    if element_symbols is not None:
        if isinstance(element_symbols, str):
            element_symbols = [element_symbols]
        df = df[df['element_symbol'].isin(element_symbols)]
    return df.copy()


def structure_to_pdb_string(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(atom_to_atom_line(row))
    return '\n'.join(lines)


def atom_to_atom_line(row):
    # Extract the columns for the line
    atom_number = row['atom_number']
    atom_name = row['atom_name']
    alt_loc = row['alt_loc']
    residue_name = row['residue_name']
    chain_id = row['chain_id']
    residue_number = row['residue_number']
    insertion_code = row['insertion_code']
    x_coord = row['x_coord']
    y_coord = row['y_coord']
    z_coord = row['z_coord']
    occupancy = row['occupancy']
    temp_factor = row['temp_factor']
    element_symbol = row['element_symbol']
    # Format the line
    line = f"{'ATOM':6}{atom_number:>5} {atom_name:<4}{alt_loc:1}{residue_name:>3} {chain_id:1}{residue_number:>4}{insertion_code:1}   "
    line += f"{x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}"
    line += f"{occupancy:>6.2f}{temp_factor:>6.2f}          "
    line += f"{element_symbol:>2}{alt_loc:1}"
    return line


class DSSRRes(object):
    def __init__(self, s):
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        i_num = 0
        for i in range(0, len(spl[1])):
            i_num = i
            try:
                cur_num = int(spl[1][i:])
                break
            except:
                continue
        self.num = int(cur_num)
        self.chain_id = spl[0]
        self.res_id = spl[1][0:i_num]


def get_motifs_from_structure(json_path):
    name = os.path.splitext(json_path.split("/")[-1])[0]
    d_out = DSSROutput(json_path)
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
    name = ""
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
