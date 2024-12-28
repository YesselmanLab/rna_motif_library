import os
import numpy as np

import pandas as pd

from rna_motif_library.classes import Hbond, Residue
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.util import calculate_dihedral_angle, get_nucleotide_atom_type


def get_closest_atoms():
    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]
    return closest_atoms


class HbondFactory:
    def __init__(self):
        self.closest_atoms = get_closest_atoms()

    def get_hbond(self, res1: Residue, res2: Residue, atom1: str, atom2: str) -> Hbond:
        atom1_coords = res1.get_atom_coords(atom1)
        atom2_coords = res2.get_atom_coords(atom2)
        if atom1_coords is None or atom2_coords is None:
            return None
        distance = np.linalg.norm(atom1_coords - atom2_coords)
        closest_atom1 = self._get_closest_atom(atom1, res1)
        closest_atom2 = self._get_closest_atom(atom2, res2)
        closest_atom1_coords = res2.get_atom_coords(closest_atom1)
        closest_atom2_coords = res1.get_atom_coords(closest_atom2)
        if closest_atom1_coords is None or closest_atom2_coords is None:
            return None
        angle = calculate_dihedral_angle(
            closest_atom1_coords, atom1_coords, atom2_coords, closest_atom2_coords
        )
        atom_type_1 = get_nucleotide_atom_type(atom1)
        atom_type_2 = get_nucleotide_atom_type(atom2)
        hbond_type = self._assign_hbond_type(res1, res2, atom_type_1, atom_type_2)
        return Hbond(
            res1.get_x3dna_residue(),
            res2.get_x3dna_residue(),
            atom1,
            atom2,
            atom_type_1,
            atom_type_2,
            distance,
            angle,
            hbond_type,
            "UNK",
        )

    def _assign_hbond_type(
        self, res1: Residue, res2: Residue, atom_type_1: str, atom_type_2: str
    ) -> str:
        if atom_type_1 == "aa" or atom_type_2 == "aa":
            return "RNA/PROTEIN"
        else:
            if atom_type_1 > atom_type_2:
                atom_type_1, atom_type_2 = atom_type_2, atom_type_1
            return f"{atom_type_1}/{atom_type_2}"

    def _get_closest_atom(self, atom_name: str, residue: Residue) -> str:
        key = f"{residue.res_id}-{atom_name}"
        if key in self.closest_atoms:
            return self.closest_atoms[key]
        return None
