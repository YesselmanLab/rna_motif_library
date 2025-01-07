import pandas as pd
import glob
import os
from typing import Dict, List, Tuple

from pydssr.dssr import DSSROutput
from pydssr.dssr_classes import DSSR_PAIR
from rna_motif_library.classes import (
    X3DNAInteraction,
    X3DNAPair,
    X3DNAResidueFactory,
    sanitize_x3dna_atom_name,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.interactions import get_x3dna_interactions


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


def get_missing_interactions(
    pairs: Dict[str, DSSR_PAIR], interactions: List[X3DNAInteraction]
):
    """Get X3DNA pairs from DSSR pairs"""
    count = 0
    for pair in pairs.values():
        res_1 = X3DNAResidueFactory.create_from_string(pair.nt1.nt_id)
        res_2 = X3DNAResidueFactory.create_from_string(pair.nt2.nt_id)
        # Parse hbond description into atom pairs and distances
        hbond_details = parse_hbond_description(pair.hbonds_desc)
        if len(hbond_details) == 0:
            continue
        atom_pairs = {}
        for atom_1, atom_2, _ in hbond_details:
            if atom_1 > atom_2:
                atom_1, atom_2 = atom_2, atom_1
            atom_pairs[atom_1 + " " + atom_2] = 0
        pair_interactions = []
        for interaction in interactions:
            if interaction.atom_1 > interaction.atom_2:
                key = interaction.atom_2 + " " + interaction.atom_1
            else:
                key = interaction.atom_1 + " " + interaction.atom_2
            if interaction.res_1 == res_1 and interaction.res_2 == res_2:
                pair_interactions.append(interaction)
                atom_pairs[key] += 1
            elif interaction.res_1 == res_2 and interaction.res_2 == res_1:
                pair_interactions.append(interaction)
                atom_pairs[key] += 1
        if len(pair_interactions) != len(hbond_details):
            print(
                pair.nt1.nt_id,
                pair.nt2.nt_id,
                len(pair_interactions),
                len(hbond_details),
            )
            count += 1

        print(count)


def main():
    count = 0
    # f = open("missing_interactions.txt", "w")
    json_files = glob.glob(os.path.join(DATA_PATH, "dssr_output/*.json"))
    for json_file in json_files:
        if "6T7T" not in json_file:
            continue
        name = os.path.basename(json_file)[:-5]
        json_path = os.path.join(DATA_PATH, "dssr_output", f"{name}.json")
        d_out = DSSROutput(json_path=json_path)
        hbonds = d_out.get_hbonds()
        interactions = get_x3dna_interactions(name, hbonds)
        get_missing_interactions(d_out.get_pairs(), interactions)
        count += 1
        print(count, name)
    # f.close()


if __name__ == "__main__":
    main()
