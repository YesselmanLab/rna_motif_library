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
canon_amino_acid_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
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
    "VAL",
]

purine_atom_names = ["C4", "N3", "C2", "N1", "C6", "C5", "N7", "C8", "N9"]
pyrimidine_atom_names = ["C4", "N3", "C2", "N1", "C6", "C5"]


atom_renames = {
    "OP1": "O1P",
    "OP2": "O2P",
    "OP3": "O3P",
}


def get_x3dna_res_id(res_id: str, num: int, chain_id: str, ins_code: str) -> str:
    if res_id[-1].isdigit():
        res_id = res_id[:-1]
    if ins_code != "":
        return f"{chain_id}.{res_id}{num}^{ins_code}"
    else:
        return f"{chain_id}.{res_id}{num}"


def sanitize_x3dna_atom_name(atom_name: str) -> str:
    if "." in atom_name:
        atom_name = atom_name.split(".")[0]
    if atom_name in atom_renames:
        atom_name = atom_renames[atom_name]

    return atom_name


def get_cif_header_str() -> str:
    s = ""
    s += "data_\n"
    s += "_entry.id test\n"
    s += "loop_\n"
    s += "_atom_site.group_PDB\n"
    s += "_atom_site.id\n"
    s += "_atom_site.auth_atom_id\n"
    s += "_atom_site.auth_comp_id\n"
    s += "_atom_site.auth_asym_id\n"
    s += "_atom_site.auth_seq_id\n"
    s += "_atom_site.pdbx_PDB_ins_code\n"
    s += "_atom_site.Cartn_x\n"
    s += "_atom_site.Cartn_y\n"
    s += "_atom_site.Cartn_z\n"
    return s
