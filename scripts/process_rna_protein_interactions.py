import os
import json
import numpy as np
import pandas as pd

from rna_motif_library.classes import (
    get_residues_from_json,
    Hbond,
    Residue,
    NucleotideAminoAcidHbond,
    X3DNAResidueFactory,
)
from rna_motif_library.settings import DATA_PATH

import numpy as np
from typing import Tuple, List, Optional


def create_nucleotide_frame(residue: Residue) -> np.ndarray:
    """
    Create a reference frame from a nucleotide residue.
    Uses C1', N9 (purines) or N1 (pyrimidines), and C4 to define the frame.

    Args:
        residue: Residue object containing atom coordinates

    Returns:
        4x4 transformation matrix where first 3x3 is rotation matrix and last column is origin
    """
    # Get required atom coordinates
    c1p_coords = residue.get_atom_coords("C1'")
    if c1p_coords is None:
        c1p_coords = residue.get_atom_coords("C1*")  # Handle alternative naming

    # Handle purine vs pyrimidine
    if residue.rtype in ["A", "G"]:  # Purines
        n_coords = residue.get_atom_coords("N9")
    else:  # Pyrimidines (C, T, U)
        n_coords = residue.get_atom_coords("N1")

    c4_coords = residue.get_atom_coords("C4")

    # Check if we have all required atoms
    if any(coord is None for coord in [c1p_coords, n_coords, c4_coords]):
        raise ValueError(
            f"Missing required atoms for residue {residue.res_id}{residue.num}"
        )

    # Convert to numpy arrays
    origin = np.array(c1p_coords)
    n_pos = np.array(n_coords)
    c4_pos = np.array(c4_coords)

    # Create coordinate system
    # Z-axis along glycosidic bond (C1' to N9/N1)
    z_axis = n_pos - origin
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Temporary y-axis toward C4
    temp_y = c4_pos - n_pos

    # X-axis perpendicular to z and temp_y
    x_axis = np.cross(z_axis, temp_y)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y-axis completes right-handed system
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin

    return transform


def get_transformed_residue(residue: Residue, transform: np.ndarray) -> Residue:
    coords_array = np.array(residue.coords)
    homog_coords = np.ones((len(coords_array), 4))
    homog_coords[:, :3] = coords_array
    transformed_coords = np.dot(homog_coords, transform.T)[:, :3]

    transformed_residue = Residue(
        chain_id=residue.chain_id,
        res_id=residue.res_id,
        num=residue.num,
        ins_code=residue.ins_code,
        rtype=residue.rtype,
        atom_names=residue.atom_names,
        coords=np.array([tuple(coord) for coord in transformed_coords]),
    )
    return transformed_residue


def align_residues(mobile: Residue, target: Residue) -> Tuple[np.ndarray, Residue]:
    """
    Align one residue to another using their reference frames.

    Args:
        mobile: Residue to be moved
        target: Target residue to align to

    Returns:
        Tuple of:
            - 4x4 transformation matrix used for alignment
            - New Residue object with transformed coordinates
    """
    # Get reference frames
    mobile_frame = create_nucleotide_frame(mobile)
    target_frame = create_nucleotide_frame(target)

    # Get transformation matrix
    transform = np.dot(np.linalg.inv(mobile_frame), target_frame)

    # Transform all coordinates
    coords_array = np.array(mobile.coords)
    homog_coords = np.ones((len(coords_array), 4))
    homog_coords[:, :3] = coords_array

    transformed_coords = np.dot(homog_coords, transform.T)[:, :3]

    # Create new residue with transformed coordinates
    aligned_residue = Residue(
        chain_id=mobile.chain_id,
        res_id=mobile.res_id,
        num=mobile.num,
        ins_code=mobile.ins_code,
        rtype=mobile.rtype,
        atom_names=mobile.atom_names,
        coords=np.array([tuple(coord) for coord in transformed_coords]),
    )

    return transform, aligned_residue


def main():
    df = pd.read_csv("rna_protein_interactions.csv")
    df = df.query(
        "res1_id == 'A' and res2_id == 'LYS' and atom1 == 'N7' and atom2 == 'NZ'"
    ).copy()
    print(len(df))
    residues = {}
    count = 0
    interactions = []
    for i, row in df.iterrows():
        if count % 100 == 0:
            print(count)
        if row["pdb_code"] not in residues:
            residues[row["pdb_code"]] = get_residues_from_json(
                os.path.join(DATA_PATH, "jsons", "residues", f"{row['pdb_code']}.json")
            )
        res1 = residues[row["pdb_code"]][row["res1"]]
        res2 = residues[row["pdb_code"]][row["res2"]]
        current_frame = create_nucleotide_frame(res1)
        transform = np.linalg.inv(current_frame)  # always to indentity / origin?
        transformed_res1 = get_transformed_residue(res1, transform)
        transformed_res2 = get_transformed_residue(res2, transform)
        res1 = X3DNAResidueFactory.create_from_string(row["res1"])
        res2 = X3DNAResidueFactory.create_from_string(row["res2"])
        hbond = Hbond(
            res1,
            res2,
            row["atom1"],
            row["atom2"],
            row["atom_type1"],
            row["atom_type2"],
            row["distance"],
            row["angle"],
            "RNA/PROTEIN",
            row["pdb_code"],
        )
        interaction = NucleotideAminoAcidHbond(
            transformed_res1, transformed_res2, hbond
        )
        interactions.append(interaction)
        count += 1
    data = [interaction.to_dict() for interaction in interactions]
    with open("rna_protein_interactions.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
