import os
import json
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

import pandas as pd
from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.residue import Residue
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.util import (
    calculate_dihedral_angle,
    calculate_angle,
    get_nucleotide_atom_type,
    sanitize_x3dna_atom_name,
    canon_amino_acid_list,
    get_cached_path,
)
from rna_motif_library.x3dna import X3DNAResidue, X3DNAInteraction, X3DNAResidueFactory


def get_closest_atoms_dict():
    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]
    return closest_atoms


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


class HbondFactory:
    def __init__(self):
        self.closest_atoms = get_closest_atoms_dict()
        self.distance_cutoff = 5.0

    def get_hbond(
        self, res1: Residue, res2: Residue, atom1: str, atom2: str, pdb_code: str
    ) -> Hbond:
        atom1_coords = res1.get_atom_coords(atom1)
        atom2_coords = res2.get_atom_coords(atom2)
        if atom1_coords is None or atom2_coords is None:
            return None
        distance = np.linalg.norm(atom1_coords - atom2_coords)
        if distance > self.distance_cutoff:
            return None
        closest_atom1 = self._get_closest_atom(atom1, res1)
        closest_atom2 = self._get_closest_atom(atom2, res2)
        closest_atom1_coords = res1.get_atom_coords(closest_atom1)
        closest_atom2_coords = res2.get_atom_coords(closest_atom2)
        if closest_atom1_coords is None or closest_atom2_coords is None:
            return None
        dihedral_angle = calculate_dihedral_angle(
            closest_atom1_coords, atom1_coords, atom2_coords, closest_atom2_coords
        )
        angle_1 = calculate_angle(closest_atom1_coords, atom1_coords, atom2_coords)
        angle_2 = calculate_angle(atom1_coords, atom2_coords, closest_atom2_coords)
        atom_type_1 = get_nucleotide_atom_type(atom1)
        atom_type_2 = get_nucleotide_atom_type(atom2)
        hbond_type = self._assign_hbond_type(atom_type_1, atom_type_2)
        return Hbond(
            res1.get_x3dna_residue(),
            res2.get_x3dna_residue(),
            atom1,
            atom2,
            atom_type_1,
            atom_type_2,
            round(distance, 2),
            round(angle_1, 2),
            round(angle_2, 2),
            round(dihedral_angle, 2),
            hbond_type,
            pdb_code,
        )

    def _assign_hbond_type(self, atom_type_1: str, atom_type_2: str) -> str:
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

        min_dist = float("inf")
        target_coords = residue.get_atom_coords(atom_name)
        closest_atom = None
        # Calculate distance to each atom
        for aname, acoords in zip(residue.atom_names, residue.coords):
            if aname == atom_name:
                continue
            dist = np.linalg.norm(target_coords - acoords)
            if dist < min_dist:
                min_dist = dist
                closest_atom = aname
        return closest_atom


def get_hbonds_from_json(json_path: str) -> List[Hbond]:
    with open(json_path) as f:
        hbonds_data = json.load(f)
        hbonds = [Hbond.from_dict(h) for h in hbonds_data]
    return hbonds


def save_hbonds_to_json(hbonds: List[Hbond], json_path: str):
    with open(json_path, "w") as f:
        json.dump([h.to_dict() for h in hbonds], f)


def score_hbond(distance, angle1, angle2, dihedral):
    """
    Calculate a score for hydrogen bond quality based on geometric criteria.
    Scores how close values are to ideal parameters:
    - Ideal distance: 2.9 Å
    - Ideal angles: 120°
    - Ideal dihedral: 0° or 180°

    Parameters:
    -----------
    distance : float
        Distance between donor and acceptor atoms in Angstroms
    angle1 : float
        First angle in degrees
    angle2 : float
        Second angle in degrees
    dihedral : float
        Dihedral angle in degrees

    Returns:
    --------
    float
        Score between 0 and 1, where 1 is ideal geometry
    """
    # Distance scoring (gaussian centered at 2.9)
    # Tighter sigma (0.3) to make it more sensitive to distance deviation
    dist_score = np.exp(-((distance - 2.9) ** 2) / (2 * 0.4**2))

    # Angle scoring (prefer 120 for both angles)
    # Using absolute deviation from 120
    angle1_score = 1 - abs(angle1 - 120) / 180
    angle2_score = 1 - abs(angle2 - 120) / 180

    # Dihedral scoring (prefer 0 or 180)
    # Take the minimum deviation from either 0 or 180
    dihedral_dev = min(abs(dihedral), abs(dihedral - 180))
    dihedral_score = 1 - dihedral_dev / 180

    # Combine scores with higher weight on distance
    total_score = (
        0.5 * dist_score
        + 0.1 * angle1_score
        + 0.1 * angle2_score
        + 0.3 * dihedral_score
    )

    return total_score


def get_cached_hbonds(pdb_id: str) -> List[Hbond]:
    json_path = get_cached_path(pdb_id, "hbonds")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Hbonds file not found for {pdb_id}")
    return get_hbonds_from_json(json_path)


# parsing x3dna stuff #################################################################


def get_hbonds_from_x3dna_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_x3dna_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
    return all_interactions


def parse_hbond_description(hbonds_desc: str) -> List[Tuple[str, str, float]]:
    """Parse DSSR hbond description into list of (atom1, atom2, distance) tuples"""
    hbond_details = []
    for hbond in hbonds_desc.split(","):
        # Extract the two atoms and distance
        if hbond.find("-") != -1:
            atoms = hbond.split("-")
        else:
            atoms = hbond.split("*")
        if len(atoms) != 2:
            continue
        # Get first atom, removing any parenthetical descriptions
        atom1 = atoms[0].split("(")[0].strip()
        # Get second atom and distance
        atom2_dist = atoms[1].split("[")
        if len(atom2_dist) != 2:
            continue
        atom2 = atom2_dist[0].split("(")[0].strip()
        distance = float(atom2_dist[1].strip("]"))
        atom1 = sanitize_x3dna_atom_name(atom1)
        atom2 = sanitize_x3dna_atom_name(atom2)
        hbond_details.append((atom1, atom2, distance))
    return hbond_details


def convert_x3dna_hbonds_to_interactions(
    hbonds: List[DSSR_HBOND],
) -> List[X3DNAInteraction]:
    # Determine type based on atom name

    rna_interactions = []
    for hbond in hbonds:
        atom_1, res_1 = hbond.atom1_id.split("@")
        atom_2, res_2 = hbond.atom2_id.split("@")
        atom_1 = sanitize_x3dna_atom_name(atom_1)
        atom_2 = sanitize_x3dna_atom_name(atom_2)
        type_1 = get_nucleotide_atom_type(atom_1)
        type_2 = get_nucleotide_atom_type(atom_2)
        for aa in canon_amino_acid_list:
            if aa in res_1:
                type_1 = "aa"
            if aa in res_2:
                type_2 = "aa"

        if type_1 == "aa" and type_2 == "aa":
            continue
        # Swap if res_1 is larger than res_2
        if res_1 > res_2:
            atom_1, atom_2 = atom_2, atom_1
            res_1, res_2 = res_2, res_1
            type_1, type_2 = type_2, type_1
        res_1 = X3DNAResidueFactory.create_from_string(res_1)
        res_2 = X3DNAResidueFactory.create_from_string(res_2)
        rna_interactions.append(
            X3DNAInteraction(
                atom_1, res_1, atom_2, res_2, float(hbond.distance), type_1, type_2
            )
        )
    return rna_interactions


def merge_hbond_interaction_data(
    rna_interactions: List[X3DNAInteraction], rnp_interactions: List[X3DNAInteraction]
) -> List[X3DNAInteraction]:
    """
    Merges RNA-RNA and RNA-protein interaction data and removes duplicates and protein-protein interactions.

    Args:
        rna_interactions (list): List of X3DNAInteraction objects from RNA-RNA interactions
        rnp_interactions (list): List of X3DNAInteraction objects from RNA-protein interactions

    Returns:
        unique_interactions (list): List of unique X3DNAInteraction objects, excluding protein-protein interactions
    """
    # Combine all interactions
    all_interactions = rna_interactions + rnp_interactions

    # Create a list of interactions, excluding protein-protein interactions
    unique_interactions = set()
    for interaction in all_interactions:
        # Skip if distance is greater than 3.3 this is not a real hbond
        if interaction.distance > 3.3:
            continue
        # Only keep interactions that aren't between two proteins
        is_protein_protein = (
            interaction.atom_type_1 == "aa" and interaction.atom_type_2 == "aa"
        )
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)
