import os
import json
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.residue import (
    Residue,
    get_cached_residues,
    are_residues_connected,
)
from rna_motif_library.settings import LIB_PATH, DATA_PATH, RESOURCES_PATH
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.logger import get_logger
from rna_motif_library.util import (
    calculate_dihedral_angle,
    calculate_angle,
    get_nucleotide_atom_type,
    sanitize_x3dna_atom_name,
    canon_amino_acid_list,
    canon_rna_res_list,
    ion_list,
    ResidueTypeAssigner,
    get_cached_path,
)
from rna_motif_library.x3dna import (
    X3DNAResidue,
    X3DNAInteraction,
    X3DNAResidueFactory,
    get_cached_dssr_output,
)

log = get_logger("hbond")


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
    res_type_1: str
    res_type_2: str
    distance: float
    angle_1: float
    angle_2: float
    dihedral_angle: float
    score: float
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
        self.distance_cutoff = 3.7
        self.closest_atoms = get_closest_atoms_dict()
        self.hbond_acceptors = json.load(
            open(os.path.join(RESOURCES_PATH, "hbond_acceptors.json"))
        )
        self.hbond_donors = json.load(
            open(os.path.join(RESOURCES_PATH, "hbond_donors.json"))
        )
        del self.hbond_acceptors["A"]["N6"]
        self.residue_type_assigner = ResidueTypeAssigner()

    def _find_hbonds_for_res_pair(
        self, res_1: Residue, res_2: Residue, pdb_name: str
    ) -> List[Hbond]:
        if (
            res_1.res_id not in self.hbond_acceptors
            or res_2.res_id not in self.hbond_donors
        ):
            return []
        potential_hbonds = []
        acceptors = self.hbond_acceptors[res_1.res_id]
        donors = self.hbond_donors[res_2.res_id]
        hbond_atom_pairs = []
        for acceptor in acceptors.keys():
            for donor in donors.keys():
                hbond_atom_pairs.append((acceptor, donor))
        for res1_atom, res2_atom in hbond_atom_pairs:
            hbond = self.get_hbond(res_1, res_2, res1_atom, res2_atom, pdb_name)
            if hbond is None:
                continue
            potential_hbonds.append(hbond)
        return potential_hbonds

    def _find_capacity_constrained_hbonds(self, unique_hbonds, res_1, res_2):
        """
        Find optimal H-bonds considering capacity constraints for donors and acceptors.
        For atoms with capacity 2, checks if the two best bonds are within 60 degrees
        using the dot product of the H-bond vectors.

        Args:
            unique_hbonds: List of Hbond objects

        Returns:
            List of selected Hbond objects that satisfy capacity constraints
        """
        # Sort hbonds by score in descending order
        unique_hbonds = sorted(unique_hbonds, key=lambda x: x.score, reverse=True)

        # Track acceptor and donor capacities
        acceptor_capacities = {}
        donor_capacities = {}
        edge_lookup = {}
        hbond_vectors = {}

        # First pass: Track capacities and calculate H-bond vectors
        for idx, hbond in enumerate(unique_hbonds):
            acceptor_key = f"{hbond.res_1.get_str()}_{hbond.atom_1}"
            donor_key = f"{hbond.res_2.get_str()}_{hbond.atom_2}"

            if acceptor_key not in acceptor_capacities:
                acceptor_capacities[acceptor_key] = self.hbond_acceptors[
                    hbond.res_1.res_id
                ][hbond.atom_1]
            if donor_key not in donor_capacities:
                donor_capacities[donor_key] = self.hbond_donors[hbond.res_2.res_id][
                    hbond.atom_2
                ]

            # Calculate H-bond vector (from acceptor to donor)
            if hbond.res_1.get_str() == res_1.get_str():
                acceptor_coords = res_1.get_atom_coords(hbond.atom_1)
                donor_coords = res_2.get_atom_coords(hbond.atom_2)
            else:
                acceptor_coords = res_2.get_atom_coords(hbond.atom_1)
                donor_coords = res_1.get_atom_coords(hbond.atom_2)
            hbond_vector = donor_coords - acceptor_coords
            hbond_vector = hbond_vector / np.linalg.norm(hbond_vector)
            hbond_vectors[idx] = hbond_vector

            edge_lookup[idx] = (hbond.score, hbond, acceptor_key, donor_key)

        # Keep track of used capacities and vectors
        used_acceptors = defaultdict(int)
        used_donors = defaultdict(int)
        acceptor_vectors = defaultdict(list)  # key -> [(idx, vector)]
        donor_vectors = defaultdict(list)  # key -> [(idx, vector)]
        selected_hbonds = []

        # Process h-bonds in order of decreasing score
        for idx in range(len(unique_hbonds)):
            if idx not in edge_lookup:
                continue

            score, hbond, acceptor_key, donor_key = edge_lookup[idx]
            hbond_vector = hbond_vectors[idx]

            # Check capacity constraints
            if used_acceptors[acceptor_key] >= acceptor_capacities[acceptor_key]:
                continue
            if used_donors[donor_key] >= donor_capacities[donor_key]:
                continue

            # For capacity 2, check angle constraints using dot product
            if (
                acceptor_capacities[acceptor_key] == 2
                and used_acceptors[acceptor_key] == 1
            ):
                prev_vector = acceptor_vectors[acceptor_key][0][1]
                angle = np.arccos(np.clip(np.dot(hbond_vector, prev_vector), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                if angle_deg < 60:
                    continue

            if donor_capacities[donor_key] == 2 and used_donors[donor_key] == 1:
                prev_vector = donor_vectors[donor_key][0][1]
                angle = np.arccos(np.clip(np.dot(hbond_vector, prev_vector), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                if angle_deg < 60:
                    continue

            selected_hbonds.append(hbond)
            used_acceptors[acceptor_key] += 1
            used_donors[donor_key] += 1
            acceptor_vectors[acceptor_key].append((idx, hbond_vector))
            donor_vectors[donor_key].append((idx, hbond_vector))

        return selected_hbonds

    def find_hbonds(self, res_1: Residue, res_2: Residue, pdb_name: str) -> List[Hbond]:
        potential_hbonds = self._find_hbonds_for_res_pair(res_1, res_2, pdb_name)
        potential_hbonds += self._find_hbonds_for_res_pair(res_2, res_1, pdb_name)
        unique_hbonds = []
        seen = []
        for hbond in potential_hbonds:
            hbond_str = f"{hbond.res_1.get_str()}_{hbond.res_2.get_str()}_{hbond.atom_1}_{hbond.atom_2}"
            hbond_str_flipped = f"{hbond.res_2.get_str()}_{hbond.res_1.get_str()}_{hbond.atom_2}_{hbond.atom_1}"
            if hbond_str in seen or hbond_str_flipped in seen:
                continue
            seen.append(hbond_str)
            seen.append(hbond_str_flipped)
            unique_hbonds.append(hbond)

        hbonds = self._find_capacity_constrained_hbonds(unique_hbonds, res_1, res_2)
        return hbonds

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

        atom_type_1 = self.residue_type_assigner.get_residue_type(
            res1.get_str(), pdb_code
        )
        atom_type_2 = self.residue_type_assigner.get_residue_type(
            res2.get_str(), pdb_code
        )
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
            score_hbond(distance, angle_1, angle_2, dihedral_angle),
            pdb_code,
        )

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


class CapacityConstrainedHbondFinder:
    def __init__(self):
        self.hbond_acceptors = json.load(
            open(os.path.join(RESOURCES_PATH, "hbond_acceptors.json"))
        )
        self.hbond_donors = json.load(
            open(os.path.join(RESOURCES_PATH, "hbond_donors.json"))
        )

    def find_hbonds(self, unique_hbonds, residues):
        """
        Find optimal H-bonds considering capacity constraints for donors and acceptors.
        For atoms with capacity 2, checks if the two best bonds are within 60 degrees
        using the dot product of the H-bond vectors.

        Args:
            unique_hbonds: List of Hbond objects
            residues: List of Residue objects containing all residues involved in H-bonds
        Returns:
            List of selected Hbond objects that satisfy capacity constraints
        """
        unique_hbonds = sorted(unique_hbonds, key=lambda x: x.score, reverse=True)

        # Initialize tracking dictionaries
        acceptor_capacities, donor_capacities = self._initialize_capacities(
            unique_hbonds
        )
        edge_lookup, hbond_vectors = self._calculate_vectors(unique_hbonds, residues)

        return self._select_hbonds(
            unique_hbonds,
            acceptor_capacities,
            donor_capacities,
            edge_lookup,
            hbond_vectors,
        )

    def _initialize_capacities(self, unique_hbonds):
        """Initialize capacity tracking dictionaries for acceptors and donors"""
        acceptor_capacities = {}
        donor_capacities = {}

        for hbond in unique_hbonds:
            acceptor_key = f"{hbond.res_1.get_str()}_{hbond.atom_1}"
            donor_key = f"{hbond.res_2.get_str()}_{hbond.atom_2}"

            if acceptor_key not in acceptor_capacities:
                acceptor_capacities[acceptor_key] = self.hbond_acceptors[
                    hbond.res_1.res_id
                ][hbond.atom_1]
            if donor_key not in donor_capacities:
                donor_capacities[donor_key] = self.hbond_donors[hbond.res_2.res_id][
                    hbond.atom_2
                ]

        return acceptor_capacities, donor_capacities

    def _calculate_vectors(self, unique_hbonds, residues):
        """Calculate H-bond vectors and create edge lookup dictionary"""
        edge_lookup = {}
        hbond_vectors = {}

        # Create lookup dict for residues
        res_lookup = {res.get_str(): res for res in residues}

        for idx, hbond in enumerate(unique_hbonds):
            acceptor_key = f"{hbond.res_1.get_str()}_{hbond.atom_1}"
            donor_key = f"{hbond.res_2.get_str()}_{hbond.atom_2}"

            # Get residues from lookup
            res1 = res_lookup[hbond.res_1.get_str()]
            res2 = res_lookup[hbond.res_2.get_str()]

            # Calculate H-bond vector (from acceptor to donor)
            acceptor_coords = res1.get_atom_coords(hbond.atom_1)
            donor_coords = res2.get_atom_coords(hbond.atom_2)

            hbond_vector = donor_coords - acceptor_coords
            hbond_vector = hbond_vector / np.linalg.norm(hbond_vector)
            hbond_vectors[idx] = hbond_vector

            edge_lookup[idx] = (hbond.score, hbond, acceptor_key, donor_key)

        return edge_lookup, hbond_vectors

    def _check_angle_constraint(self, vector1, vector2):
        """Check if angle between two vectors satisfies the 60 degree constraint"""
        angle = np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))
        angle_deg = np.degrees(angle)
        return angle_deg >= 60

    def _select_hbonds(
        self,
        unique_hbonds,
        acceptor_capacities,
        donor_capacities,
        edge_lookup,
        hbond_vectors,
    ):
        """Select H-bonds that satisfy capacity and angle constraints"""
        used_acceptors = defaultdict(int)
        used_donors = defaultdict(int)
        acceptor_vectors = defaultdict(list)
        donor_vectors = defaultdict(list)
        selected_hbonds = []

        for idx in range(len(unique_hbonds)):
            if idx not in edge_lookup:
                continue

            score, hbond, acceptor_key, donor_key = edge_lookup[idx]
            hbond_vector = hbond_vectors[idx]

            # Check capacity constraints
            if used_acceptors[acceptor_key] >= acceptor_capacities[acceptor_key]:
                continue
            if used_donors[donor_key] >= donor_capacities[donor_key]:
                continue

            # Check angle constraints for capacity 2
            if (
                acceptor_capacities[acceptor_key] == 2
                and used_acceptors[acceptor_key] == 1
            ):
                if not self._check_angle_constraint(
                    hbond_vector, acceptor_vectors[acceptor_key][0][1]
                ):
                    continue

            if donor_capacities[donor_key] == 2 and used_donors[donor_key] == 1:
                if not self._check_angle_constraint(
                    hbond_vector, donor_vectors[donor_key][0][1]
                ):
                    continue

            selected_hbonds.append(hbond)
            used_acceptors[acceptor_key] += 1
            used_donors[donor_key] += 1
            acceptor_vectors[acceptor_key].append((idx, hbond_vector))
            donor_vectors[donor_key].append((idx, hbond_vector))

        return selected_hbonds


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


def get_flipped_hbond(hbond: Hbond) -> Hbond:
    flipped_hbond = Hbond(
        hbond.res_2,
        hbond.res_1,
        hbond.atom_2,
        hbond.atom_1,
        hbond.res_type_2,
        hbond.res_type_1,
        hbond.distance,
        hbond.angle_2,
        hbond.angle_1,
        hbond.dihedral_angle,
        hbond.score,
        hbond.pdb_name,
    )
    return flipped_hbond


# parsing x3dna stuff #################################################################


def generate_hbonds(pdb_name: str) -> List[Hbond]:
    residues = get_cached_residues(pdb_name)
    hf = HbondFactory()
    hbonds = []
    res_coms = {}
    seen = []
    for res in residues.values():
        res_coms[res.get_str()] = res.get_center_of_mass()
    data = []
    for i, r1 in enumerate(residues.values()):
        if r1.res_id not in canon_rna_res_list:
            continue
        res_1_com = res_coms[r1.get_str()]
        for j, r2 in enumerate(residues.values()):
            if r1.res_id == r2.res_id:
                continue
            res_2_com = res_coms[r2.get_str()]
            com_dist = np.linalg.norm(res_1_com - res_2_com)
            if com_dist > 15.0:
                continue
            if f"{r2.get_str()}_{r1.get_str()}" in seen:
                continue
            if f"{r1.get_str()}_{r2.get_str()}" in seen:
                continue
            seen.append(f"{r1.get_str()}_{r2.get_str()}")
            seen.append(f"{r2.get_str()}_{r1.get_str()}")
            hbonds += hf.find_hbonds(r1, r2, pdb_name)

    cchf = CapacityConstrainedHbondFinder()
    hbonds = cchf.find_hbonds(hbonds, residues.values())
    final_hbonds = []
    for hbond in hbonds:
        if hbond.res_type_1 != "RNA":
            hbond = get_flipped_hbond(hbond)
        final_hbonds.append(hbond)
        h_data = hbond.to_dict()
        h_data["res_1"] = hbond.res_1.get_str()
        h_data["res_2"] = hbond.res_2.get_str()
        data.append(h_data)
    if len(data) == 0:
        f = open(os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_name}.csv"), "w")
        f.write("res_1,res_2,atom_1,atom_2,res_type_1,res_type_2,distance,angle_1,angle_2,dihedral_angle,score,pdb_name\n")
        f.close()
        return final_hbonds

    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_name}.csv"), index=False
    )
    return final_hbonds


def generate_hbonds_from_x3dna(pdb_name: str) -> List[Hbond]:
    residues = get_cached_residues(pdb_name)
    dssr_output = get_cached_dssr_output(pdb_name)
    dssr_hbonds = dssr_output.get_hbonds()
    all_dssr_hbonds = get_hbonds_from_x3dna_interactions(pdb_name, dssr_hbonds)
    hf = HbondFactory()
    df = pd.read_json(
        os.path.join(
            LIB_PATH, "rna_motif_library", "resources", "small_molecule_instances.json"
        )
    )
    df = df.query("pdb_id == @pdb_name")
    small_molecules_res_ids = df["res_id"].unique()
    hbonds = []
    data = []
    for xhbond in all_dssr_hbonds:
        try:
            hbond = hf.get_hbond(
                residues[xhbond.res_1.get_str()],
                residues[xhbond.res_2.get_str()],
                xhbond.atom_1,
                xhbond.atom_2,
                pdb_name,
            )
        except KeyError:
            log.error(
                f"KeyError for {xhbond.res_1.get_x3dna_str()} {xhbond.res_2.get_x3dna_str()}"
            )
            log.error(
                f"KeyError for {xhbond.res_1.get_str()} {xhbond.res_2.get_str()} {pdb_name}"
            )
            continue
        if hbond is None:
            continue
        if hbond.res_1.get_str() in small_molecules_res_ids:
            hbond.res_type_1 = "small_molecule"
        if hbond.res_2.get_str() in small_molecules_res_ids:
            hbond.res_type_2 = "small_molecule"
        # rna should always be res_1
        if hbond.res_type_1 == "aa" or hbond.res_type_1 == "small_molecule":
            hbond = get_flipped_hbond(hbond)
        hbonds.append(hbond)
        h_data = hbond.to_dict()
        h_data["res_1"] = hbond.res_1.get_str()
        h_data["res_2"] = hbond.res_2.get_str()
        data.append(h_data)
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_name}.csv"), index=False
    )
    return hbonds


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
        # Skip if distance is greater than 3.5 this is not a real hbond
        if interaction.distance > 3.5:
            continue
        # Only keep interactions that aren't between two proteins
        is_protein_protein = (
            interaction.atom_type_1 == "aa" and interaction.atom_type_2 == "aa"
        )
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)
