import os
from typing import List, Tuple, Dict
import json
import pandas as pd
import numpy as np

from pydssr.dssr_classes import DSSR_HBOND, DSSR_PAIR
from pydssr.dssr import DSSROutput

from rna_motif_library.classes import (
    X3DNAInteraction,
    Hbond,
    Basepair,
    Residue,
    X3DNAResidue,
    X3DNAResidueFactory,
    X3DNAPair,
    canon_amino_acid_list,
    sanitize_x3dna_atom_name,
)
from rna_motif_library.logger import get_logger
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.util import calculate_dihedral_angle, get_nucleotide_atom_type

log = get_logger("interactions")


# top level function ###################################################################


def get_hbonds_from_json(json_path: str) -> List[Hbond]:
    with open(json_path) as f:
        hbonds_data = json.load(f)
        hbonds = [Hbond.from_dict(h) for h in hbonds_data]
    return hbonds


def get_basepairs_from_json(json_path: str) -> List[Basepair]:
    with open(json_path) as f:
        basepairs_data = json.load(f)
        basepairs = [Basepair.from_dict(bp) for bp in basepairs_data]
    return basepairs


def get_hbonds_and_basepairs(
    pdb_name: str, overwrite: bool = False
) -> Tuple[List[Hbond], List[Basepair]]:
    # Check if hbonds and basepairs json files already exist
    hbonds_json_path = os.path.join(DATA_PATH, "jsons", "hbonds", f"{pdb_name}.json")
    basepairs_json_path = os.path.join(
        DATA_PATH, "jsons", "basepairs", f"{pdb_name}.json"
    )

    if (
        os.path.exists(hbonds_json_path)
        and os.path.exists(basepairs_json_path)
        and not overwrite
    ):
        log.info(f"Loading existing hbonds and basepairs for {pdb_name}")
        hbonds = get_hbonds_from_json(hbonds_json_path)
        basepairs = get_basepairs_from_json(basepairs_json_path)
        return hbonds, basepairs
    log.info(f"Generating hbonds and basepairs for {pdb_name}")
    json_path = os.path.join(DATA_PATH, "dssr_output", f"{pdb_name}.json")
    residue_data = json.loads(
        open(os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_name}.json")).read()
    )
    residues = {k: Residue.from_dict(v) for k, v in residue_data.items()}
    dssr_output = DSSROutput(json_path=json_path)
    hbonds = dssr_output.get_hbonds()
    pairs = dssr_output.get_pairs()
    save_basepair_params(pairs, pdb_name)
    x3dna_interactions = get_x3dna_interactions(pdb_name, hbonds)
    x3dna_pairs = get_x3dna_pairs(pairs, x3dna_interactions)
    hbonds = get_hbonds(pdb_name, residues, x3dna_interactions, x3dna_pairs)
    basepairs = get_pairs(pdb_name, hbonds, x3dna_pairs)
    return hbonds, basepairs


# handles converting X3DNA to current objects ##########################################


def save_basepair_params(pairs: List[DSSR_PAIR], pdb_name: str):
    all_data = []

    for p in pairs.values():
        e = p.bp.split("-")
        if len(e) != 2:
            e = p.bp.split("+")
        if len(e) != 2:
            e = [p.bp[0], p.bp[-1]]
        if e[0] > e[1]:
            bp_type = e[1] + e[0]
        else:
            bp_type = e[0] + e[1]
        data = {
            "res_1": p.nt1.nt_id,
            "res_2": p.nt2.nt_id,
            "bp_type": bp_type,
            "bp_name": p.name,
            "lw": p.LW,
            "hbonds_desc": p.hbonds_desc,
            "frame": np.array(
                [
                    p.frame["x_axis"],
                    p.frame["y_axis"],
                    p.frame["z_axis"],
                ]
            ),
            "origin": p.frame["origin"],
            "shear": p.bp_params[0],
            "stretch": p.bp_params[1],
            "stagger": p.bp_params[2],
            "buckle": p.bp_params[3],
            "propeller": p.bp_params[4],
            "opening": p.bp_params[5],
        }
        all_data.append(data)
    df = pd.DataFrame(all_data)
    df.to_json(
        os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_name}.json"),
        orient="records",
    )


def get_x3dna_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_x3dna_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
    return all_interactions


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
        # Skip if distance is greater than 3.3 this is not a real hbond
        if distance > 3.3:
            continue
        atom1 = sanitize_x3dna_atom_name(atom1)
        atom2 = sanitize_x3dna_atom_name(atom2)
        hbond_details.append((atom1, atom2, distance))
    return hbond_details


def get_x3dna_pairs(pairs: Dict[str, DSSR_PAIR], interactions: List[X3DNAInteraction]):
    """Get X3DNA pairs from DSSR pairs"""
    x3dna_pairs = []
    for pair in pairs.values():
        res_1 = X3DNAResidueFactory.create_from_string(pair.nt1.nt_id)
        res_2 = X3DNAResidueFactory.create_from_string(pair.nt2.nt_id)
        # Parse hbond description into atom pairs and distances
        hbond_details = parse_hbond_description(pair.hbonds_desc)
        pair_interactions = []
        for interaction in interactions:
            if interaction.res_1 == res_1 and interaction.res_2 == res_2:
                pair_interactions.append(interaction)
            elif interaction.res_1 == res_2 and interaction.res_2 == res_1:
                pair_interactions.append(interaction)
        x3dna_pairs.append(
            X3DNAPair(res_1, res_2, pair_interactions, pair.name, pair.LW)
        )
    return x3dna_pairs


# handles converting to final objects ##################################################


def get_closest_atoms():
    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]
    return closest_atoms


def get_atom_coords(
    atom_name: str, residue: X3DNAResidue, residues: Dict[str, Residue]
) -> Tuple[float, float, float]:
    res = residues.get(residue.get_str())
    if res is None:
        return None
    return res.get_atom_coords(atom_name)


def find_closest_atom(
    atom_name: str,
    residue: X3DNAResidue,
    residues: Dict[str, Residue],
    closest_atoms: Dict[str, str],
) -> str:
    # First try looking up in dictionary
    key = f"{residue.res_id}-{atom_name}"
    if key in closest_atoms:
        return closest_atoms[key]

    # If not in dictionary, calculate distances to all atoms in residue
    target_coords = get_atom_coords(atom_name, residue, residues)
    if target_coords is None:
        return None

    min_dist = float("inf")
    closest_atom = None
    res = residues.get(residue.get_str())

    # Calculate distance to each atom
    for aname, acoords in zip(res.atom_names, res.coords):
        if aname == atom_name:
            continue
        dist = np.linalg.norm(target_coords - acoords)
        if dist < min_dist:
            min_dist = dist
            closest_atom = aname
    return closest_atom


def get_hbonds(
    pdb_name: str,
    residues: Dict[str, Residue],
    x3dna_interactions: List[X3DNAInteraction],
    x3dna_pairs: List[X3DNAPair],
) -> List[Hbond]:
    hbonds = []
    pair_dict = {}
    for pair in x3dna_pairs:
        pair_dict[pair.res_1.get_str() + "-" + pair.res_2.get_str()] = pair
        pair_dict[pair.res_2.get_str() + "-" + pair.res_1.get_str()] = pair
    closest_atoms = get_closest_atoms()
    for interaction in x3dna_interactions:
        closest_atom_1 = find_closest_atom(
            interaction.atom_1, interaction.res_1, residues, closest_atoms
        )
        closest_atom_2 = find_closest_atom(
            interaction.atom_2, interaction.res_2, residues, closest_atoms
        )
        closest_atom_1_coords = get_atom_coords(
            closest_atom_1, interaction.res_1, residues
        )
        closest_atom_2_coords = get_atom_coords(
            closest_atom_2, interaction.res_2, residues
        )
        atom_1_coords = get_atom_coords(interaction.atom_1, interaction.res_1, residues)
        atom_2_coords = get_atom_coords(interaction.atom_2, interaction.res_2, residues)
        if (
            closest_atom_1_coords is None
            or closest_atom_2_coords is None
            or atom_1_coords is None
            or atom_2_coords is None
        ):
            dihedral_angle = None
        else:
            dihedral_angle = calculate_dihedral_angle(
                closest_atom_1_coords,
                atom_1_coords,
                atom_2_coords,
                closest_atom_2_coords,
            )
        hbond_type = ""
        if interaction.res_1.get_str() + "-" + interaction.res_2.get_str() in pair_dict:
            pair = pair_dict[
                interaction.res_1.get_str() + "-" + interaction.res_2.get_str()
            ]
            if pair.bp_type == "WC":
                hbond_type = "BP-WC"
            else:
                hbond_type = "BP-NON-WC"
        elif interaction.atom_type_1 == "aa" or interaction.atom_type_2 == "aa":
            hbond_type = "RNA/PROTEIN"
        else:
            atom_1_type = interaction.atom_type_1
            atom_2_type = interaction.atom_type_2
            if atom_1_type > atom_2_type:
                atom_1_type, atom_2_type = atom_2_type, atom_1_type
            hbond_type = f"{atom_1_type}/{atom_2_type}"
        hbond_info = Hbond(
            interaction.res_1,
            interaction.res_2,
            interaction.atom_1,
            interaction.atom_2,
            interaction.atom_type_1,
            interaction.atom_type_2,
            interaction.distance,
            dihedral_angle,
            hbond_type,
            pdb_name,
        )
        hbonds.append(hbond_info)
    return hbonds


def get_pairs(
    pdb_name: str,
    hbonds: List[Hbond],
    x3dna_pairs: List[X3DNAPair],
) -> List[Basepair]:
    pairs = []
    for pair in x3dna_pairs:
        pair_hbonds = []
        for interaction in pair.interactions:
            for hbond in hbonds:
                if (
                    hbond.res_1 == interaction.res_1
                    and hbond.res_2 == interaction.res_2
                    and hbond.atom_1 == interaction.atom_1
                    and hbond.atom_2 == interaction.atom_2
                ):
                    pair_hbonds.append(hbond)
        if len(pair_hbonds) != len(pair.interactions):
            print("Mismatch in number of hbonds and interactions")
            print(len(pair_hbonds), len(pair.interactions))
        pairs.append(
            Basepair(
                pair.res_1,
                pair.res_2,
                pair_hbonds,
                pair.bp_type,
                pair.bp_name,
                pdb_name,
            )
        )
    return pairs
