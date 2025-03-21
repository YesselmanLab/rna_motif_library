import numpy as np
import glob
import os
from typing import Optional
import pandas as pd


from rna_motif_library.settings import DATA_PATH

canon_rna_res_list = ["A", "U", "G", "C"]

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

canon_res_list = canon_rna_res_list + canon_amino_acid_list

ion_list = [
    "V",
    "CR",
    "2HP",
    "3CO",
    "AG",
    "AL",
    "AU",
    "BA",
    "BEF",
    "BR",
    "CA",
    "CD",
    "CL",
    "CS",
    "CO",
    "CU",
    "F",
    "FE",
    "FE2",
    "HG",
    "IOD",
    "IR",
    "IR3",
    "K",
    "LI",
    "LU",
    "MG",
    "MN",
    "NA",
    "NH4",
    "NI",
    "OS",
    "PB",
    "PT",
    "SM",
    "SO4",
    "SR",
    "TB",
    "TL",
    "ZN",
]

purine_atom_names = ["C4", "N3", "C2", "N1", "C6", "C5", "N7", "C8", "N9"]
pyrimidine_atom_names = ["C4", "N3", "C2", "N1", "C6", "C5"]

wc_basepairs = ["A-U", "U-A", "G-C", "C-G"]
wc_basepairs_w_gu = wc_basepairs + ["G-U", "U-G"]

atom_renames = {
    "OP1": "O1P",
    "OP2": "O2P",
    "OP3": "O3P",
}


def get_cached_path(pdb_id: str, name: str) -> str:
    if name == "dssr_output":
        return os.path.join(DATA_PATH, "dssr_output", f"{pdb_id}.json")
    else:
        return os.path.join(DATA_PATH, "jsons", name, f"{pdb_id}.json")


def get_pdb_ids(
    pdb_id: Optional[str] = None,
    directory: Optional[str] = None,
    pdb_list: Optional[list] = None,
    csv_path: Optional[str] = None,
) -> list:
    """
    Get list of PDB codes based on input parameters.

    Args:
        pdb_id (str, optional): Single PDB id to process. Defaults to None.
        directory (str, optional): Directory containing PDB files. Defaults to None.

    Returns:
        list: List of PDB ids to process
    """
    pdb_ids = []
    if pdb_id is not None:
        pdb_ids.append(pdb_id)
    elif pdb_list is not None:
        pdb_ids = pdb_list
    elif directory is not None:
        pdb_ids = [os.path.basename(file)[:-4] for file in os.listdir(directory)]
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
        pdb_ids = df["pdb_id"].tolist()
    else:
        files = glob.glob(os.path.join(DATA_PATH, "pdbs", "*.cif"))
        for file in files:
            pdb_id = os.path.basename(file)[:-4]
            pdb_ids.append(pdb_id)
    return pdb_ids


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


class CifParser:
    def __init__(self):
        self.loops = {}
        self.df = None

    def _find_loops(self, lines: list) -> None:
        """Find all loop_ sections and their fields in the CIF file."""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                i += 1
                continue
            # Found a loop
            if line == "loop_":
                loop_fields = []
                i += 1
                # Collect all fields that start with underscore
                while i < len(lines) and lines[i].strip().startswith("_"):
                    field = lines[i].strip()
                    loop_fields.append(field)
                    i += 1
                # Get first data line after field names
                # Save all data until next loop
                first_data = None
                data_lines = []
                j = i
                while j < len(lines):
                    line = lines[j].strip()
                    if not line or line.startswith("#"):
                        j += 1
                        continue
                    if line == "loop_" or line.startswith("#"):
                        break
                    if first_data is None:
                        first_data = line
                    data_lines.append(line)
                    j += 1
                # Determine loop type from the first field
                if loop_fields:
                    loop_type = loop_fields[0].split(".")[0].replace("_", "")
                    self.loops[loop_type] = {
                        "fields": loop_fields,
                        "first_data": first_data,
                        "data_lines": data_lines,
                    }
            else:
                i += 1

    def _get_atom_loop_fields(self, file_path: str) -> list:
        """Get the field names for the atom site loop."""
        required_fields = [
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ]

        data_loop = None
        for key, loop in self.loops.items():
            first_data = loop["first_data"]
            fields = loop["fields"]
            if first_data and first_data.startswith("ATOM"):
                # Check if all required coordinate fields are present
                if not all(
                    any(field.endswith(coord) for field in fields)
                    for coord in required_fields
                ):
                    raise ValueError(
                        f"Missing required coordinate fields in {file_path}"
                    )
                data_loop = loop

        if data_loop is not None:
            return data_loop

        for key, loop in self.loops.items():
            fields = loop["fields"]
            contains = 0
            for field in fields:
                for required_field in required_fields:
                    if field.find(required_field) != -1:
                        contains += 1
            if contains >= len(required_fields):
                data_loop = loop
                break
        if data_loop is None:
            raise ValueError(f"No atom records found in {file_path}")

        return data_loop

    def _parse_atom_records(self, data_loop: dict) -> pd.DataFrame:
        """Parse atom records into a DataFrame."""
        atoms = []
        for line in data_loop["data_lines"]:
            values = [v.strip('"') for v in line.split()]
            atoms.append(values)

        if not atoms:
            raise ValueError("No valid atom records found")

        return pd.DataFrame(atoms, columns=data_loop["fields"])

    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse a CIF file and return a DataFrame of atom records.

        Args:
            file_path: Path to the CIF file

        Returns:
            pd.DataFrame containing the atom records
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        self._find_loops(lines)
        data_loop = self._get_atom_loop_fields(file_path)
        # Remove '_atom_site.' prefix from field names
        data_loop["fields"] = [name.split(".")[-1] for name in data_loop["fields"]]
        fields = []
        for field in data_loop["fields"]:
            if field.endswith("Cartn_x"):
                fields.append("Cartn_x")
            elif field.endswith("Cartn_y"):
                fields.append("Cartn_y")
            elif field.endswith("Cartn_z"):
                fields.append("Cartn_z")
            else:
                fields.append(field)
        data_loop["fields"] = fields
        self.df = self._parse_atom_records(data_loop)
        # Convert coordinate columns to float
        # self.df[["Cartn_x", "Cartn_y", "Cartn_z"]] = self.df[
        #    ["Cartn_x", "Cartn_y", "Cartn_z"]
        # ].astype(float)

        return self.df


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


class ResidueTypeAssigner:
    def __init__(self):
        df = pd.read_csv(
            os.path.join(DATA_PATH, "ligands", "single_type_res_identities.csv")
        )
        self.single_type_res_identities = {}
        for _, row in df.iterrows():
            self.single_type_res_identities[row["res_id"]] = row["type"]
        df = pd.read_csv(
            os.path.join(DATA_PATH, "ligands", "multi_type_res_identities.csv")
        )
        self.multi_type_res_identities = {}
        for _, row in df.iterrows():
            self.multi_type_res_identities[row["res_str"] + row["pdb_id"]] = row["type"]
        self.dna = ["DA", "DC", "DT", "DG"]

    def get_residue_type(self, res_str: str, pdb_id: str) -> str:
        res_id = res_str.split("-")[1]
        key = res_str + pdb_id
        if res_id in canon_rna_res_list:
            return "RNA"
        elif res_id in canon_amino_acid_list:
            return "PROTEIN"
        elif res_id in self.dna:
            return "DNA"
        elif res_id in ion_list:
            return "ION"
        # check exceptions first
        elif key in self.multi_type_res_identities:
            return self.multi_type_res_identities[key]
        elif res_id in self.single_type_res_identities:
            return self.single_type_res_identities[res_id]
        else:
            return "UNKNOWN"


def get_non_redundant_sets(csv_path: str) -> list:
    d = {}
    df = pd.read_csv(csv_path, header=None, names=["set_id", "repr_id", "all_ids"])
    for _, row in df.iterrows():
        repr_id = row["repr_id"].split("|")[0]
        all_ids = row["all_ids"].split(",")
        all_pdb_ids = [id.split("|")[0] for id in all_ids]
        d[repr_id] = all_pdb_ids
    return d
