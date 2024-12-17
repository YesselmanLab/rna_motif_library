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
    "VAL"
]
solvent_res = [
    "HOH",  # water
    "MG",  # magnesium
    "ZN",  # zinc
    "IUM",  # iodine ion
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
residue_reclassifier = {
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "T",
    "DU": "U"
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
    res_1: X3DNAResidue
    res_2: X3DNAResidue
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


class Residue:
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
        if ins_code is None:
            ins_code = ""
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
        if not isinstance(other, Residue):
            return False

        return self.is_equal(other)

    def is_equal(self, other, check_coords=False):
        if check_coords:
            return (
                self.chain_id == other.chain_id
                and self.res_id == other.res_id
                and self.num == other.num
                and self.ins_code == other.ins_code
                and self.rtype == other.rtype
                and np.allclose(self.coords, other.coords)
            )
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

    def to_cif_str(self, acount=0):
        s = ""
        # Write the data from the DataFrame
        ins_code = self.ins_code
        if ins_code == "":
            ins_code = "?"
        for atom_name, coord in zip(self.atom_names, self.coords):
            s += (
                f"{'ATOM':<8}"
                f"{str(acount):<7}"
                f"{str(atom_name):<6}"
                f"{str(self.res_id):<6}"
                f"{str(self.num):<6}"
                f"{str(self.chain_id):<6}"
                f"{str(ins_code):<6}"
                f"{str(coord[0]):<12}"
                f"{str(coord[1]):<12}"
                f"{str(coord[2]):<12}\n"
            )
            acount += 1

        return s, acount


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
