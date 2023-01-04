import os
from pydssr import dssr
import atomium
from atomium.pdb import Ligand, Residue
from atomium.mmcif import structure_to_mmcif_string

def structure_to_pdb_string(structure):
    """Converts a :py:class:`.AtomStructure` to a .pdb filestring.

    :param AtomStructure structure: the structure to convert.
    :rtype: ``str``"""

    lines = []
    pack_sequences(structure, lines)
    atoms = sorted(structure.atoms(), key=lambda a: a.id)
    for i, atom in enumerate(atoms):
        atom_to_atom_line(atom, lines)
        if isinstance(atom.het, Residue) and (
         atom is atoms[-1] or atoms[i + 1].chain is not atom.chain or
          isinstance(atoms[i + 1].het, Ligand)):
            last = lines[-1]
            lines.append(f"TER   {last[6:11]}      {last[17:20]} {last[21]}{last[22:26]}{last[26]}")
    return "\n".join(lines)


def pack_sequences(structure, lines):
    """Adds SEQRES lines from polymer sequence data.

    :param AtomStructure structure: the structure to convert.
    :param list lines: the string lines to update."""

    try:
        for chain in sorted(structure.chains(), key=lambda c: c.id):
            residues = valerius.from_string(chain.sequence).codes
            length = len(residues)
            line_count = ceil(length / 13)
            for line_num in range(line_count):
                lines += ["SEQRES {:>3} {} {:>4}  {}".format(
                 line_num + 1, chain.id, length,
                 " ".join(residues[line_num * 13: (line_num + 1) * 13])
                )]
    except AttributeError: pass


def atom_to_atom_line(a, lines):
    """Converts an :py:class:`.Atom` to an ATOM or HETATM record. ANISOU lines
    will also be added where appropriate.

    :param Atom a: The Atom to pack.
    :param list lines: the string lines to update."""

    line = "{:6}{:5} {:4} {:3} {:1}{:4}{:1}   "
    line += "{:>8}{:>8}{:>8}  1.00{:6}          {:>2}{:2}"
    id_, residue_name, chain_id, residue_id, insert_code = "", "", "", "", ""
    if a.het:
        id_, residue_name = a.het.id, a.het._name
        chain_id = a.chain.id if a.chain is not None else ""
        if len(str(chain_id)) > 1:
            chain_id = str(chain_id)[0]
        residue_id = int("".join([c for c in id_ if c.isdigit() or c == "-"]))
        if residue_id > 10000:
            residue_id = int(str(residue_id)[1:])
        insert_code = id_[-1] if id_ and id_[-1].isalpha() else ""
    atom_name = a._name or ""
    atom_name = " " + atom_name if len(atom_name) < 4 else atom_name
    line = line.format(
     "HETATM" if isinstance(a.het, Ligand) or a._is_hetatm else "ATOM",
     len(lines)+1, atom_name, residue_name, chain_id, residue_id, insert_code,
     "{:.3f}".format(a.location[0]) if a.location[0] is not None else "",
     "{:.3f}".format(a.location[1]) if a.location[1] is not None else "",
     "{:.3f}".format(a.location[2]) if a.location[2] is not None else "",
     "{:.2f}".format(a.bvalue).strip().rjust(6) if a.bvalue is not None else "",
     a.element or "", str(int(a.charge))[::-1] if a.charge else "",
    )
    lines.append(line)
    if a.anisotropy != [0, 0, 0, 0, 0, 0]:
        lines.append(atom_to_anisou_line(a, atom_name,
         residue_name, chain_id, residue_id, insert_code))


class DSSRRes(object):
    def __init__(self, s):
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        for i in range(0, len(spl[1])):
            try:
                cur_num = int(spl[1][i:])
                break
            except:
                continue
        self.num = int(cur_num)
        self.chain_id = spl[0]
        self.res_id = spl[1][0:i]


def get_motifs_from_structure(json_path):
    name = os.path.splitext(json_path.split("/")[-1])[0]
    d_out = dssr.DSSROutput(json_path=json_path)
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
        'sugar:base': 0, 'sugar:sugar' : 0, 'sugar:phos': 0, 'phos:base': 0, 'phos:sugar' : 0,
        'phos:phos' : 0, 'base:aa'   : 0, 'sugar:aa': 0, 'phos:aa': 0
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


def write_res_coords_to_pdb(nts, pdb_model, pdb_path):
    res = []
    for nt in nts:
        r = DSSRRes(nt)
        new_nt = r.chain_id + "." + str(r.num)
        atom_res = pdb_model.model.residue(new_nt)
        if atom_res is None:
            continue
        res.append(atom_res)
    s = ""
    for r in res:
        lines = structure_to_pdb_string(r).split("\n")
        s += "\n".join(lines[:-1]) + "\n"
    f = open(f"{pdb_path}.pdb", "w")
    f.write(s)
    f.close()


def write_motif_coords_to_pdbs(motifs, pdb_file):
    pdb_model = atomium.open(pdb_file)
    count = 0
    for m in motifs:
        res = []
        for nt in m.nts_long:
            spl = nt.split(".")
            new_nt = spl[0] + "." + spl[1][1:]
            res.append(pdb_model.model.residue(new_nt))
        s = ""
        for r in res:
            lines = structure_to_pdb_string(r).split("\n")
            s += "\n".join(lines[:-1]) + "\n"
        f = open(f"{m.mtype}.{count}.pdb", "w")
        count += 1
        f.write(s)
        f.close()
