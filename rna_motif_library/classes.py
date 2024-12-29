import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from rna_motif_library.util import *


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
    angle_1: float
    angle_2: float
    dihedral_angle: float
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
class BasepairParameters:
    shear: float
    stretch: float
    stagger: float
    buckle: float
    propeller: float
    opening: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasepairParameters":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass(frozen=True, order=True)
class Basepair:
    res_1: X3DNAResidue
    res_2: X3DNAResidue
    hbonds: List[Hbond]
    bp_type: str
    lw: str
    pdb_name: str
    hbond_score: float
    bp_params: BasepairParameters

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
        data["bp_params"] = BasepairParameters.from_dict(data["bp_params"])
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
        data["bp_params"] = self.bp_params.to_dict()
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

    def get_base_atom_coords(self) -> List[Tuple[float, float, float]]:
        """
        Get coordinates of all base atoms (non-sugar/phosphate atoms).

        Returns:
            List[Tuple[float, float, float]]: List of (x,y,z) coordinates for base atoms
        """
        base_coords = []
        for atom_name, coord in zip(self.atom_names, self.coords):
            # Skip sugar and phosphate atoms
            if atom_name.startswith(
                (
                    "C1'",
                    "C2'",
                    "C3'",
                    "C4'",
                    "C5'",
                    "O2'",
                    "O3'",
                    "O4'",
                    "O5'",
                    "P",
                    "O1P",
                    "O2P",
                    "O3P",
                )
            ):
                continue
            base_coords.append(coord)
        return base_coords

    def get_x3dna_str(self):
        res_id = self.res_id
        if self.res_id[-1].isdigit():
            res_id = res_id[:-1]
        if self.ins_code != "":
            return f"{self.chain_id}.{res_id}{self.num}^{self.ins_code}"
        else:
            return f"{self.chain_id}.{res_id}{self.num}"

    def get_x3dna_residue(self) -> X3DNAResidue:
        return X3DNAResidue(
            self.chain_id, self.res_id, self.num, self.ins_code, self.rtype
        )

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

    def to_cif_str(self, acount=1):
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
                f"{str(round(coord[0], 3)):<12}"
                f"{str(round(coord[1], 3)):<12}"
                f"{str(round(coord[2], 3)):<12}\n"
            )
            acount += 1

        return s, acount

    def to_cif(self, file_path: str):
        s = get_cif_header_str()
        cif_str, _ = self.to_cif_str()
        s += cif_str
        with open(file_path, "w") as f:
            f.write(s)
        f.close()


def get_residues_from_json(json_path: str) -> List[Residue]:
    with open(json_path, "r") as f:
        residue_data = json.load(f)
    return {k: Residue.from_dict(v) for k, v in residue_data.items()}


class NucleotideAminoAcidHbond:
    def __init__(self, res_1: Residue, res_2: Residue, hbond: Hbond):
        self.res_1 = res_1
        self.res_2 = res_2
        self.hbond = hbond

    @classmethod
    def from_dict(cls, data: dict):
        data["res_1"] = Residue.from_dict(data["res_1"])
        data["res_2"] = Residue.from_dict(data["res_2"])
        data["hbond"] = Hbond.from_dict(data["hbond"])
        return cls(**data)

    def to_dict(self):
        return {
            "res_1": self.res_1.to_dict(),
            "res_2": self.res_2.to_dict(),
            "hbond": self.hbond.to_dict(),
        }

    def to_cif_str(self):
        s = ""
        acount = 1
        res_1_str, acount = self.res_1.to_cif_str(acount)
        res_2_str, acount = self.res_2.to_cif_str(acount)
        s += res_1_str
        s += res_2_str
        return s

    def to_cif(self, file_path: str):
        s = get_cif_header_str()
        cif_str = self.to_cif_str()
        s += cif_str
        with open(file_path, "w") as f:
            f.write(s)
        f.close()
