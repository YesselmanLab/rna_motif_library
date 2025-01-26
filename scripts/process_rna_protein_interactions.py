import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rna_motif_library.motif import (
    Hbond,
    Residue,
)
from rna_motif_library.residue import get_residues_from_json
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.x3dna import X3DNAResidueFactory

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


def generate_summary(df):
    """
    Generate summary statistics for RNA-protein interactions.

    Args:
        df: DataFrame containing interaction data

    Returns:
        DataFrame with summary statistics grouped by residue and atom pairs
    """
    # Group and calculate summary statistics using agg
    summary = (
        df.groupby(["res1_id", "res2_id", "atom1", "atom2"])
        .agg(
            count=("distance", "size"),
            distance_mean=("distance", "mean"),
            distance_std=("distance", "std"),
            dihedral_mean=("dihedral_angle", "mean"),
            dihedral_std=("dihedral_angle", "std"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
        )
        .reset_index()
    )

    return summary


def main():
    df = pd.read_csv("rna_protein_interactions.csv")
    nucs = ["A", "C", "G", "U"]
    df = df.query("res1_id.isin(@nucs)").copy()
    df_sum = generate_summary(df)


if __name__ == "__main__":
    main()
