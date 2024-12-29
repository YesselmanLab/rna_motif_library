import numpy as np

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


def get_nucleotide_atom_type(atom_name: str) -> str:
    """Get the type of nucleotide atom (phosphate, sugar, or base).

    Args:
        atom_name: The atom name to check.

    Returns:
        str: The type of atom - either 'phos', 'sugar', or 'base'.
    """
    if atom_name.startswith(("P", "O1P", "O2P", "O3P", "OP1", "OP2", "OP3")):
        return "phos"
    elif atom_name.startswith(
        ("O2'", "O3'", "O4'", "O5'", "C1'", "C2'", "C3'", "C4'", "C5'")
    ):
        return "sugar"
    else:
        return "base"


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


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle between three points in 3D space.

    Args:
        p1 (np.ndarray): Coordinates of first point
        p2 (np.ndarray): Coordinates of second point (vertex)
        p3 (np.ndarray): Coordinates of third point

    Returns:
        float: Angle in degrees
    """
    # Calculate vectors from vertex to points
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Calculate cosine of angle
    cos_angle = dot_product / (mag1 * mag2)

    # Handle numerical errors
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    # Convert to degrees
    angle = np.arccos(cos_angle)
    angle_deg = np.degrees(angle)

    return round(angle_deg, 1)


def calculate_dihedral_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """
    Calculate the dihedral angle between 4 points in 3D space.

    Args:
        p1 (np.ndarray): Coordinates of first point
        p2 (np.ndarray): Coordinates of second point
        p3 (np.ndarray): Coordinates of third point
        p4 (np.ndarray): Coordinates of fourth point

    Returns:
        float: Dihedral angle in degrees
    """
    # Calculate vectors between points
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    # Calculate normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # Normalize normal vectors
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    # Calculate angle between normal vectors
    cos_angle = np.dot(n1, n2)
    # Handle numerical errors
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = np.arccos(cos_angle)
    # Determine sign of angle
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return round(np.degrees(angle), 1)
