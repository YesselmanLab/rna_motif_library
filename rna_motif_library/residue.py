import json
import os
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif

from rna_motif_library.x3dna import X3DNAResidue, X3DNAResidueFactory, get_residue_type
from rna_motif_library.util import (
    sanitize_x3dna_atom_name,
    get_x3dna_res_id,
    get_cif_header_str,
    get_cached_path,
    CifParser,
)


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

    def get_str(self):
        return f"{self.chain_id}-{self.res_id}-{self.num}-{self.ins_code}"

    def __hash__(self):
        """Make Residue hashable by using immutable attributes."""
        return hash((self.chain_id, self.res_id, self.num, self.ins_code))

    def __eq__(self, other):
        """Define equality for hash comparison."""
        if not isinstance(other, Residue):
            return False
        return self.is_equal(other)

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

    def get_sugar_atom_coords(self) -> List[Tuple[float, float, float]]:
        """
        Get coordinates of all sugar atoms (C1', C2', C3', C4', C5', O2', O3', O4', O5').

        Returns:
            List[Tuple[float, float, float]]: List of (x,y,z) coordinates for sugar atoms
        """
        sugar_coords = []
        for atom_name, coord in zip(self.atom_names, self.coords):
            if atom_name.startswith(
                ("C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'")
            ):
                sugar_coords.append(coord)
        return sugar_coords

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
                and np.allclose(self.coords, other.coords, atol=0.01)
            )
        return (
            self.chain_id == other.chain_id
            and self.res_id == other.res_id
            and self.num == other.num
            and self.ins_code == other.ins_code
            and self.rtype == other.rtype
        )

    def move(self, vector: Tuple[float, float, float]):
        """
        Moves the residue by adding the given vector to all atom coordinates.

        Args:
            vector (Tuple[float, float, float]): 3D vector to add to coordinates
        """
        self.coords = [
            (x + vector[0], y + vector[1], z + vector[2]) for x, y, z in self.coords
        ]

    def get_center_of_mass(self) -> np.ndarray:
        """
        Computes and returns the center of mass of the residue.

        Returns:
            np.ndarray: The x,y,z coordinates of the center of mass as a numpy array
        """
        coords = np.array(self.coords)
        return np.mean(coords, axis=0)

    # to different file formats #######################################################

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

        return s

    def to_cif(self, file_path: str):
        s = get_cif_header_str()
        s += self.to_cif_str()
        with open(file_path, "w") as f:
            f.write(s)
        f.close()

    def to_pdb_str(self):
        """
        Creates a PDB format string representation of the residue.

        Returns:
            str: PDB formatted string
        """
        s = ""
        for i, (atom_name, coord) in enumerate(zip(self.atom_names, self.coords), 1):
            # Pad atom name with spaces according to PDB format
            if len(atom_name) < 4:
                atom_name = f" {atom_name:<3}"
            else:
                atom_name = f"{atom_name:<4}"

            # Format insertion code
            ins_code = self.ins_code if self.ins_code else " "
            chain_id = self.chain_id
            if len(chain_id) > 1:
                chain_id = chain_id[0]
            s += (
                f"ATOM  {i:5d} {atom_name} {self.res_id:<3} {chain_id}"
                f"{self.num:4d}{ins_code:1}   "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           {atom_name[0]:>2}\n"
            )
        return s


def get_cached_residues(pdb_id: str) -> Dict[str, Residue]:
    json_path = get_cached_path(pdb_id, "residues")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Residues file not found for {pdb_id}")
    return get_residues_from_json(json_path)


def save_residues_to_json(residues: Dict[str, Residue], json_path: str):
    with open(json_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in residues.items()}, f)


def get_residues_from_json(json_path: str) -> Dict[str, Residue]:
    with open(json_path, "r") as f:
        residue_data = json.load(f)
    return {k: Residue.from_dict(v) for k, v in residue_data.items()}


def get_residues_from_pdb(pdb_path: str) -> Dict[str, Residue]:
    ppdb = PandasPdb().read_pdb(pdb_path)
    df_atom = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
    residues = {}
    for i, g in df_atom.groupby(
        ["chain_id", "residue_number", "residue_name", "insertion"]
    ):
        coords = g[["x_coord", "y_coord", "z_coord"]].values
        atom_names = g["atom_name"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        chain_id, res_num, res_name, ins_code = i
        if ins_code == "None" or ins_code is None:
            ins_code = ""
        x3dna_res_id = get_x3dna_res_id(res_name, res_num, chain_id, ins_code)
        x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
        residues[x3dna_res_id] = Residue.from_x3dna_residue(
            x3dna_res, atom_names, coords
        )
    return residues


def get_residues_from_cif(cif_path: str) -> Dict[str, Residue]:
    try:
        ppdb = PandasMmcif().read_mmcif(cif_path)
        df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
    except Exception as e:
        parser = CifParser()
        df = parser.parse(cif_path)
    residues = {}
    for i, g in df.groupby(
        ["auth_asym_id", "auth_seq_id", "auth_comp_id", "pdbx_PDB_ins_code"]
    ):
        coords = g[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        atom_names = g["auth_atom_id"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        chain_id, res_num, res_name, ins_code = i
        if ins_code == "?":
            ins_code = ""
        x3dna_res = X3DNAResidue(
            chain_id, res_name, res_num, ins_code, get_residue_type(res_name)
        )
        residues[x3dna_res.get_str()] = Residue.from_x3dna_residue(
            x3dna_res, atom_names, coords
        )
    return residues
