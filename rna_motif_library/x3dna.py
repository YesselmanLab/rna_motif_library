import os
from typing import List, Tuple
from dataclasses import dataclass

from pydssr.dssr import DSSROutput

from rna_motif_library.util import canon_amino_acid_list, get_cached_path
from rna_motif_library.settings import DATA_PATH


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
        return f"{self.chain_id}-{self.res_id}-{self.num}-{self.ins_code}"

    def get_x3dna_str(self):
        res_id = self.res_id
        if self.res_id[-1].isdigit():
            res_id = res_id[:-1]
        if self.ins_code != "":
            return f"{self.chain_id}.{res_id}{self.num}^{self.ins_code}"
        else:
            return f"{self.chain_id}.{res_id}{self.num}"

    def __str__(self):
        return self.get_str()


def get_residue_type(res_id: str) -> str:
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


def get_cached_dssr_output(pdb_id: str) -> DSSROutput:
    json_path = get_cached_path(pdb_id, "dssr_output")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"DSSR output file not found for {pdb_id}")
    return DSSROutput(json_path=json_path)
