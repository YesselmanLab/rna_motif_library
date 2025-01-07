from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import os

from rna_motif_library.classes import (
    Residue,
    get_residues_from_json,
    get_residues_from_pdb,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import canon_res_list, get_pdb_codes


def find_connected_atoms(
    atom_coord: np.ndarray,
    all_coords: List[np.ndarray],
    all_names: List[str],
    max_bond_length: float = 1.6,
) -> List[Tuple[str, np.ndarray]]:
    """
    Find atoms that are likely bonded to the given atom based on distance.
    """
    connected = []
    for other_coord, other_name in zip(all_coords, all_names):
        if np.array_equal(atom_coord, other_coord):
            continue
        dist = np.linalg.norm(atom_coord - other_coord)
        if dist <= max_bond_length:
            connected.append((other_name, other_coord))
    return connected


def identify_potential_sites(residue: Residue) -> Tuple[List[str], List[str]]:
    """
    Identifies potential hydrogen bond donors and acceptors in a nucleotide residue
    based on geometric analysis, atom types, and presence of hydrogen atoms.

    Args:
        residue (Residue): Residue object containing atom names and coordinates

    Returns:
        Tuple[List[str], List[str]]: Lists of donor and acceptor atom names
    """
    donors = []
    acceptors = []

    # Convert coordinates to numpy arrays
    coords = [np.array(coord) for coord in residue.coords]

    # Get all base atoms (excluding sugar/phosphate)
    base_indices = [
        i
        for i, name in enumerate(residue.atom_names)
        if not name.startswith(
            (
                "C2'",
                "C3'",
                "C4'",
                "C5'",
                "P",
                "O3'",
                "O4'",
                "O1P",
                "O2P",
                "O3P",
            )
        )
    ]

    potential_atoms = [(residue.atom_names[i], coords[i]) for i in base_indices]

    for i, (atom_name, coord) in enumerate(potential_atoms):
        # Only consider N and O atoms as potential donors/acceptors
        if not atom_name.startswith(("N", "O")):
            continue
        # Find connected atoms
        connected = find_connected_atoms(
            coord, [c for _, c in potential_atoms], [n for n, _ in potential_atoms]
        )
        if not connected:
            continue
        # Check for connected hydrogens
        has_hydrogen = any(n.startswith("H") for n, _ in connected)

        if atom_name.startswith("N"):
            if has_hydrogen and len(connected) == 3:  # N-H group - donor
                donors.append(atom_name)
            elif len(connected) == 2:  # sp2 N without H - acceptor
                acceptors.append(atom_name)

        elif atom_name.startswith("O"):
            if has_hydrogen:  # O-H group - donor
                donors.append(atom_name)
            acceptors.append(atom_name)

    return donors, acceptors


def generate_residues_with_hydrogen():
    pdb_codes = get_pdb_codes()
    seen = []

    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
        residues = get_residues_from_json(json_path)
        for residue in residues.values():
            if residue.res_id in seen:
                continue
            seen.append(residue.res_id)
            s = residue.to_pdb_str()
            with open("test.pdb", "w") as f:
                f.write(s)
            os.system(
                f"reduce -BUILD test.pdb > data/residues_w_h_pdbs/{residue.res_id}.pdb"
            )


def main():
    pdb_codes = get_pdb_codes()
    seen = []
    acceptor_donor_data = []
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
        residues = get_residues_from_json(json_path)
        print(pdb_code)
        for residue in residues.values():
            if residue.res_id in seen:
                continue
            seen.append(residue.res_id)
            try:
                h_residues = get_residues_from_pdb(
                    os.path.join(
                        DATA_PATH, "residues_w_h_pdbs", f"{residue.res_id}.pdb"
                    )
                )
            except:
                print(f"No h-residue found for {residue.res_id}")
                continue
            h_residue = list(h_residues.values())[0]
            donors, acceptors = identify_potential_sites(h_residue)
            acceptor_donor_data.append(
                {
                    "residue_id": residue.res_id,
                    "donors": donors,
                    "acceptors": acceptors,
                }
            )

    df = pd.DataFrame(acceptor_donor_data)
    df.to_json("hbond_acceptor_and_donors.json", orient="records")


if __name__ == "__main__":
    main()
