import pandas as pd
import numpy as np
import json
from dataclasses import dataclass

from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS
from typing import Dict, List, Any, Tuple

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


class PandasMmcifOverride(PandasMmcif):
    """
    Class to override standard behavior for handling mmCIF files in Pandas,
    particularly to address inconsistencies between ATOM and HETATM records.

    """

    def _construct_df(self, text: str) -> pd.DataFrame:
        """
        Constructs a DataFrame from mmCIF text.

        Args:
            text (str): The mmCIF file content as a string.

        Returns:
            combined_df (pd.DataFrame): A combined DataFrame of ATOM and HETATM records.

        """
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data

        df: Dict[str, pd.DataFrame] = {}
        full_df = pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        # Combine ATOM and HETATM records into a single DataFrame
        combined_df = pd.DataFrame(
            full_df[(full_df.group_PDB == "ATOM") | (full_df.group_PDB == "HETATM")]
        )

        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)

        return combined_df  # Return the combined DataFrame


@dataclass(frozen=True, order=True)
class X3DNAResidue:
    chain_id: str
    res_id: str
    num: int
    ins_code: str
    rtype: str

    @classmethod
    def from_dict(cls, d):
        """Create X3DNAResidue from dictionary"""
        return cls(**d)

    def to_dict(self):
        """Convert X3DNAResidue to dictionary"""
        return vars(self)

    def get_str(self):
        res_id = self.res_id
        if self.res_id[-1].isdigit():
            res_id = res_id[:-1]
        if self.ins_code != "":
            return f"{self.chain_id}.{res_id}{self.num}^{self.ins_code}"
        else:
            return f"{self.chain_id}.{res_id}{self.num}"

    def __str__(self):
        return self.get_str()


class X3DNAResidueFactory:
    """Factory class for creating X3DNAResidue objects from X3DNA notation strings."""

    @classmethod
    def create_from_string(cls, s: str) -> X3DNAResidue:
        """
        Creates a X3DNAResidue object from a X3DNA residue notation string.

        Args:
            s (str): Given residue (something like "C.G1515" or "C.G1515^A")

        Returns:
            X3DNAResidue: A X3DNAResidue object containing the parsed chain ID, residue ID, number and insertion code
        """
        # Handle insertion code if present
        ins_code = ""
        if "^" in s:
            s, ins_code = s.split("^")

        spl = s.split(".")
        res_id, num = cls._split_at_trailing_numbers(spl[1])
        # this is a negative number
        if res_id.endswith("-"):
            num = -num
            res_id = res_id[:-1]
        chain_id = spl[0]
        rtype = cls._get_residue_type(res_id)
        return X3DNAResidue(chain_id, res_id, num, ins_code, rtype)

    @staticmethod
    def _split_at_trailing_numbers(s: str) -> Tuple[str, int]:
        """
        Splits a string at the longest sequence of numbers at the end.

        Args:
            s (str): Input string to split

        Returns:
            tuple[str, int]: A tuple containing (non-numeric prefix, numeric suffix).
                            If no trailing numbers found, returns (original string, 0)
        """
        # Find the longest numeric sequence at the end
        numeric_suffix = ""
        prefix = s

        # Work backwards from end of string
        i = len(s) - 1
        while i >= 0 and s[i].isdigit():
            numeric_suffix = s[i] + numeric_suffix
            prefix = s[:i]
            i -= 1

        return prefix, int(numeric_suffix)

    @staticmethod
    def _get_residue_type(res_id: str) -> str:
        """
        Determines the type of residue from its ID.

        Args:
            res_id (str): Residue identifier (e.g. "A", "DA", "GLY")

        Returns:
            str: Type of residue ("rna", "dna", "aa", or "unknown")
        """
        if res_id in ["A", "C", "G", "U"]:
            return "rna"
        elif res_id in ["DA", "DC", "DG", "DT"]:
            return "dna"
        elif res_id in canon_amino_acid_list:
            return "aa"
        else:
            return "unknown"


@dataclass(frozen=True, order=True)
class X3DNAInteraction:
    """
    Class to represent an X3DNA interaction.
    """

    atom_1: str
    res_1: X3DNAResidue
    atom_2: str
    res_2: X3DNAResidue
    distance: float
    atom_type_1: str
    atom_type_2: str


@dataclass(frozen=True, order=True)
class X3DNAPair:
    res_1: str
    res_2: str
    interactions: List[X3DNAInteraction]
    bp_type: str
    bp_name: str


@dataclass(frozen=True, order=True)
class Hbond:
    res_1: X3DNAResidue
    res_2: X3DNAResidue
    atom_1: str
    atom_2: str
    atom_type_1: str
    atom_type_2: str
    distance: float
    angle: float
    hbond_type: str
    pdb_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hbond":
        """
        Creates a HbondInfo instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing HbondInfo attributes

        Returns:
            HbondInfo: New HbondInfo instance
        """
        # Convert X3DNAResidue objects
        data["res_1"] = X3DNAResidue.from_dict(data["res_1"])
        data["res_2"] = X3DNAResidue.from_dict(data["res_2"])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts HbondInfo instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing HbondInfo attributes
        """
        data = vars(self).copy()
        # Convert X3DNAResidue objects to dicts
        data["res_1"] = self.res_1.to_dict()
        data["res_2"] = self.res_2.to_dict()
        return data


@dataclass(frozen=True, order=True)
class Basepair:
    res_1: X3DNAInteraction
    res_2: X3DNAInteraction
    hbonds: List[Hbond]
    bp_type: str
    bp_name: str
    pdb_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Basepair":
        """
        Creates a Basepair instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing Basepair attributes

        Returns:
            Basepair: New Basepair instance
        """
        # Convert X3DNAInteraction objects
        data["res_1"] = X3DNAResidue.from_dict(data["res_1"])
        data["res_2"] = X3DNAResidue.from_dict(data["res_2"])
        # Convert list of Hbond objects
        data["hbonds"] = [Hbond.from_dict(hbond) for hbond in data["hbonds"]]
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts Pair instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing Basepair attributes
        """
        data = vars(self).copy()
        data["res_1"] = self.res_1.to_dict()
        data["res_2"] = self.res_2.to_dict()
        data["hbonds"] = [hbond.to_dict() for hbond in self.hbonds]
        return data


class PotentialTertiaryContact:
    def __init__(
        self,
        motif_1: str,
        motif_2: str,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
    ):
        """
        Holds data about potential tertiary contacts.
        Purpose is to get data ready for export to CSV and tertiary contacts.
        Some data will be left out.

        Args:


        """
        self.motif_1 = motif_1
        self.motif_2 = motif_2
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle


class SingleMotifInteraction:

    def __init__(
        self,
        motif_name: str,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
    ) -> None:
        """
        Holds data for H-bond interactions within a single motif.
        Purpose is to get data ready for export to CSV.
        Therefore, some data from the HBondInteraction class will be left out.

        Args:
            motif_name (str): name of motif the interaction comes from
            res_1 (str): residue 1 in the interaction
            res_2 (str): residue 2 in the interaction
            atom_1 (str): atom 1 in the interaction
            atom_2 (str): atom 2 in the interaction
            type_1 (str): component of residue 1 in interaction (base/sugar/phos/aa)
            type_2 (str): component of residue 2 in interaction (base/sugar/phos/aa)
            distance (float): distance of interaction, in angstroms
            angle (float): dihedral angle of interaction, in degrees

        """
        self.motif_name = motif_name
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle


class HBondInteraction:
    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
        pdb: pd.DataFrame,
        first_atom_df: pd.DataFrame,
        second_atom_df: pd.DataFrame,
        third_atom_df: pd.DataFrame,
        fourth_atom_df: pd.DataFrame,
        pdb_name: str,
    ) -> None:
        """
        Holds data for H-bond interaction.
        Used to store all the data about interactions.

        Args:
            res_1 (str): residue 1 in the interaction
            res_2 (str): residue 2 in the interaction
            atom_1 (str): atom 1 in the interaction
            atom_2 (str): atom 2 in the interaction
            type_1 (str): residue type 1 in the interaction
            type_2 (str): residue type 2 in the interaction
            distance (float): distance between atoms in interaction
            angle (float): dihedral angle between two residues
            pdb (pd.DataFrame): interaction PDB
            first_atom_df (pd.DataFrame): PDB of the first atom in the interaction
            second_atom_df (pd.DataFrame): PDB of the second atom in the interaction
            third_atom_df (pd.DataFrame): PDB of the third atom connected to the first atom
            fourth_atom_df (pd.DataFrame): PDB of the fourth atom connected to the second atom
            pdb_name (str): name of PDB interaction comes from

        """
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle
        self.pdb = pdb
        self.first_atom_df = first_atom_df
        self.second_atom_df = second_atom_df
        self.third_atom_df = third_atom_df
        self.fourth_atom_df = fourth_atom_df
        self.pdb_name = pdb_name


class HBondInteractionFactory:
    """
    Intermediate class to assist in building complete HBondInteraction data.

    Args:
        res_1 (str): residue 1 ID
        res_2 (str): residue 2 ID
        atom_1 (str): atom 1 ID
        atom_2 (str): atom 2 ID
        distance (float): distance between interacting atoms (in angstroms)
        residue_pair (str): pair comparing the types as returned by DSSR
        quality (str): quality of h-bond

    """

    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        distance: float,
        residue_pair: str,
        quality: str,
    ) -> None:
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.distance = distance
        self.residue_pair = residue_pair
        self.quality = quality


class Motif:
    """
    Class to hold motif data. This data is final and should not be changed once built.
    """

    def __init__(
        self,
        motif_name: str,
        motif_type: str,
        pdb: str,
        size: str,
        sequence: str = None,
        res_list: List[str] = None,
        strands: Any = None,
        motif_pdb: pd.DataFrame = None,
    ) -> None:
        """
        Initialize a Motif object

        Args:
            motif_name (str): Name of the motif
            motif_type (str): Motif type
            pdb (str): PDB where motif is found
            size (str): Size of motif; reflects the structure and means different things depending on the type of motif
            sequence (str): Sequence in motif
            res_list (list): List of residues in the motif
            strands (list): List of strands in motif
            motif_pdb (pd.DataFrame): PDB data of the motif

        Returns:
            None
        """
        self.motif_name = motif_name
        self.motif_type = motif_type
        self.pdb = pdb
        self.size = size
        self.sequence = sequence
        self.res_list = res_list if res_list is not None else []
        self.strands = strands if strands is not None else []
        self.motif_pdb = motif_pdb if motif_pdb is not None else pd.DataFrame()

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a Motif object from a dictionary.

        Args:
            data (dict): Dictionary containing motif data

        Returns:
            Motif: A new Motif object with the data from the dictionary
        """

        # Convert motif_pdb back to DataFrame if it exists
        if data["motif_pdb"] is not None:
            data["motif_pdb"] = pd.DataFrame.from_records(data["motif_pdb"])

        # Convert strands back to Residue objects if they exist
        if data["strands"]:
            data["strands"] = [
                [Residue.from_dict(res) for res in strand] for strand in data["strands"]
            ]

        return cls(
            motif_name=data["motif_name"],
            motif_type=data["motif_type"],
            pdb=data["pdb"],
            size=data["size"],
            sequence=data["sequence"],
            res_list=data["res_list"],
            strands=data["strands"],
            motif_pdb=data["motif_pdb"],
        )

    def __eq__(self, other):
        """
        Check if two Motif objects are equal by comparing their dictionary representations.

        Args:
            other: Another Motif object to compare with

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if not isinstance(other, type(self)):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """
        Returns the object as a dictionary.

        Returns:
            The entire instance of a Motif class inside a dictionary.
            Intended for writing the motif to JSON.
        """
        return {
            "motif_name": self.motif_name,
            "motif_type": self.motif_type,
            "pdb": self.pdb,
            "size": self.size,
            "sequence": self.sequence,
            "res_list": self.res_list,
            "strands": [[res.to_dict() for res in strand] for strand in self.strands],
            "motif_pdb": (
                self.motif_pdb.to_dict(orient="records")
                if isinstance(self.motif_pdb, pd.DataFrame) and not self.motif_pdb.empty
                else None
            ),
        }

    def to_json(self, filepath: str) -> None:
        """
        Saves the motif object as a JSON file.

        Args:
            filepath (str): Path where the JSON file should be saved

        Returns:
            None
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)


def get_motifs_from_json(json_path: str) -> List[Motif]:
    """
    Get motifs from a JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return [Motif.from_dict(d) for d in data]


class Residue:
    """
    Class to hold data on individual residues, used for building strands and sequences in find_strands

    Args:
        chain_id (str): chain ID
        res_id (str): residue ID
        ins_code (str): ID code, sometimes used instead of residue ID, often is None
        mol_name (str): molecule name
        pdb (pd.DataFrame): DataFrame to hold the actual contents of the residue obtained from the .cif file
    """

    def __init__(
        self,
        chain_id: str,
        res_id: str,
        ins_code: str,
        mol_name: str,
        pdb: pd.DataFrame,
    ) -> None:
        self.chain_id = chain_id
        self.res_id = res_id
        self.ins_code = ins_code
        self.mol_name = mol_name
        self.pdb = pdb

    @classmethod
    def from_dict(cls, data: dict):
        data["pdb"] = pd.DataFrame.from_records(data["pdb"])
        return cls(**data)

    def __eq__(self, other):
        """
        Checks if two Residue objects are equal by comparing all their attributes.

        Args:
            other: Another Residue object to compare with

        Returns:
            bool: True if the residues are equal, False otherwise
        """
        if not isinstance(other, Residue):
            return False

        return (
            self.chain_id == other.chain_id
            and self.res_id == other.res_id
            and self.ins_code == other.ins_code
            and self.mol_name == other.mol_name
            and self.pdb.equals(other.pdb)
        )

    def to_dict(self) -> dict:
        """
        Converts residue information to a dictionary.

        Returns:
            dict: Dictionary containing residue attributes including:
                - chain_id: Chain identifier
                - res_id: Residue identifier
                - ins_code: Insertion code
                - mol_name: Molecule name
                - pdb: PDB data as records or None if empty
        """
        return {
            "chain_id": self.chain_id,
            "res_id": self.res_id,
            "ins_code": self.ins_code,
            "mol_name": self.mol_name,
            "pdb": self.pdb.to_dict(orient="records"),
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)

    def to_pdb_str(self):
        return "PDB_STR"


class ResidueNew:

    def __init__(
        self,
        chain_id: str,
        res_id: str,
        num: int,
        ins_code: str,
        rtype: str,
        atom_names: List[str],
        coords: List[Tuple[float, float, float]],
    ) -> None:
        self.chain_id = chain_id
        self.res_id = res_id
        self.num = num
        self.ins_code = ins_code
        self.rtype = rtype
        self.atom_names = atom_names
        self.coords = coords

    @classmethod
    def from_x3dna_residue(
        cls,
        x3dna_res: X3DNAResidue,
        atom_names: List[str],
        coords: List[Tuple[float, float, float]],
    ):
        return cls(
            x3dna_res.chain_id,
            x3dna_res.res_id,
            x3dna_res.num,
            x3dna_res.ins_code,
            x3dna_res.rtype,
            atom_names,
            coords,
        )

    @classmethod
    def from_dict(cls, data: dict):
        data["coords"] = np.array(data["coords"])
        return cls(**data)

    def get_atom_coords(self, atom_name: str) -> Tuple[float, float, float]:
        if atom_name not in self.atom_names:
            return None
        return self.coords[self.atom_names.index(atom_name)]

    def get_x3dna_str(self):
        res_id = self.res_id
        if self.res_id[-1].isdigit():
            res_id = res_id[:-1]
        if self.ins_code != "":
            return f"{self.chain_id}.{res_id}{self.num}^{self.ins_code}"
        else:
            return f"{self.chain_id}.{res_id}{self.num}"

    def __eq__(self, other):
        """
        Checks if two Residue objects are equal by comparing all their attributes.

        Args:
            other: Another Residue object to compare with

        Returns:
            bool: True if the residues are equal, False otherwise
        """
        if not isinstance(other, ResidueNew):
            return False

        return (
            self.chain_id == other.chain_id
            and self.res_id == other.res_id
            and self.num == other.num
            and self.ins_code == other.ins_code
            and self.rtype == other.rtype
        )

    def to_dict(self) -> dict:
        """
        Converts residue information to a dictionary.

        Returns:
            dict: Dictionary containing residue attributes including:
                - chain_id: Chain identifier
                - res_id: Residue identifier
                - num: Residue number
                - ins_code: Insertion code
                - rtype: Residue type
                - atom_names: List of atom names
                - coords: List of atom coordinates
        """
        return {
            "chain_id": self.chain_id,
            "res_id": self.res_id,
            "num": self.num,
            "ins_code": self.ins_code,
            "rtype": self.rtype,
            "atom_names": self.atom_names,
            "coords": self.coords.tolist(),
        }

    def to_cif_str(self):
        pass


def extract_longest_numeric_sequence(input_string: str) -> str:
    """
    Extracts the longest numeric sequence from a given string.

    Args:
        input_string (str): The string to extract the numeric sequence from.

    Returns:
        longest_sequence (str): The longest numeric sequence found in the input string.

    """
    longest_sequence = ""
    current_sequence = ""
    for c in input_string:
        if c.isdigit() or (
            c == "-" and (not current_sequence or current_sequence[0] == "-")
        ):
            current_sequence += c
            if len(current_sequence) >= len(longest_sequence):
                longest_sequence = current_sequence
        else:
            current_sequence = ""
    return longest_sequence
