import os
import json
import numpy as np
import pandas as pd

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.resources import Residue
from rna_motif_library.resources import Residue
from rna_motif_library.util import purine_atom_names, pyrimidine_atom_names
import numpy as np
from typing import Tuple, List, Optional


class NucleotideReferenceFrameGenerator:
    def __init__(self):
        self.ideal_bases = load_ideal_bases()
        self.ideal_bases["DA"] = self.ideal_bases["A"]
        self.ideal_bases["DC"] = self.ideal_bases["C"]
        self.ideal_bases["DG"] = self.ideal_bases["G"]
        self.ideal_bases["DU"] = self.ideal_bases["U"]

    def get_reference_frame(self, residue: Residue) -> np.ndarray:
        base_atoms = self._get_base_atoms(residue)
        fitted_xyz, R, orgi, rms_value = self.generate(
            self._get_base_atoms(self.ideal_bases[residue.res_id]), base_atoms
        )
        return R

    def generate(self, sxyz, exyz):
        """
        Perform least-squares fitting between two structures. To generate a rotation
        matrix that will be used as a reference frame for a given nucleotide. This is
        ported from the open source x3dna code func: ls_fitting

        Parameters:
        -----------
        sxyz : numpy.ndarray
            Source coordinates (structure to be fitted), shape (n, 3)
        exyz : numpy.ndarray
            Target coordinates (reference structure), shape (n, 3)

        Returns:
        --------
        fitted_xyz : numpy.ndarray
            Fitted coordinates, shape (n, 3)
        R : numpy.ndarray
            Rotation matrix, shape (3, 3)
        orgi : numpy.ndarray
            Translation vector, shape (3,)
        rms_value : float
            Root mean square deviation between fitted and target structures
        """

        n = len(sxyz)
        if n < 3:
            raise ValueError("Too few atoms for least-squares fitting")

        # Get the covariance matrix U
        def cov_matrix(x, y):
            xm = x - x.mean(axis=0)
            ym = y - y.mean(axis=0)
            return np.dot(xm.T, ym)

        U = cov_matrix(sxyz, exyz)

        # Construct 4x4 symmetric matrix N
        N = np.zeros((4, 4))
        N[0, 0] = U[0, 0] + U[1, 1] + U[2, 2]
        N[1, 1] = U[0, 0] - U[1, 1] - U[2, 2]
        N[2, 2] = -U[0, 0] + U[1, 1] - U[2, 2]
        N[3, 3] = -U[0, 0] - U[1, 1] + U[2, 2]

        N[0, 1] = N[1, 0] = U[1, 2] - U[2, 1]
        N[0, 2] = N[2, 0] = U[2, 0] - U[0, 2]
        N[0, 3] = N[3, 0] = U[0, 1] - U[1, 0]
        N[1, 2] = N[2, 1] = U[0, 1] + U[1, 0]
        N[1, 3] = N[3, 1] = U[2, 0] + U[0, 2]
        N[2, 3] = N[3, 2] = U[1, 2] + U[2, 1]

        # Get eigenvalues and eigenvectors of N
        D, V = np.linalg.eigh(N)

        # Get rotation matrix from eigenvector with largest eigenvalue
        q = V[:, -1]  # quaternion
        N = np.outer(q, q)

        R = np.zeros((3, 3))
        R[0, 0] = N[0, 0] + N[1, 1] - N[2, 2] - N[3, 3]
        R[0, 1] = 2 * (N[1, 2] - N[0, 3])
        R[0, 2] = 2 * (N[1, 3] + N[0, 2])
        R[1, 0] = 2 * (N[2, 1] + N[0, 3])
        R[1, 1] = N[0, 0] - N[1, 1] + N[2, 2] - N[3, 3]
        R[1, 2] = 2 * (N[2, 3] - N[0, 1])
        R[2, 0] = 2 * (N[3, 1] - N[0, 2])
        R[2, 1] = 2 * (N[3, 2] + N[0, 1])
        R[2, 2] = N[0, 0] - N[1, 1] - N[2, 2] + N[3, 3]

        # Calculate centroids
        ave_sxyz = np.mean(sxyz, axis=0)
        ave_exyz = np.mean(exyz, axis=0)

        # Calculate translation vector
        orgi = ave_exyz - np.dot(ave_sxyz, R)

        # Calculate fitted coordinates
        fitted_xyz = np.dot(sxyz, R) + orgi

        # Calculate RMS deviation
        diff = fitted_xyz - exyz
        rms_value = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

        return fitted_xyz, R, orgi, rms_value

    def _get_base_atoms(self, residue: Residue) -> np.ndarray:
        base_atoms = []
        if residue.res_id in ["A", "G", "DA", "DG"]:
            base_atoms = purine_atom_names
        elif residue.res_id in ["U", "C", "DC", "DG"]:
            base_atoms = pyrimidine_atom_names
        else:
            raise ValueError(f"Unknown residue: {residue.res_id}")
        coords = []
        for atom in base_atoms:
            coords.append(residue.coords[residue.atom_names.index(atom)])
        return np.array(coords)


def create_basepair_frame(res1: Residue, res2: Residue) -> np.ndarray:
    """
    Create a reference frame from a base pair.
    Works with both canonical and non-canonical base pairs.
    Frame is centered between the bases with Y-axis along the pseudo-dyad axis.

    Args:
        res1: First residue of the base pair
        res2: Second residue of the base pair

    Returns:
        4x4 transformation matrix where first 3x3 is rotation matrix and last column is origin
    """

    # Get glycosidic nitrogen positions (N9 for purines, N1 for pyrimidines)
    def get_glyco_N(res: Residue) -> np.ndarray:
        if res.rtype in ["A", "G"]:  # Purines
            n_coords = res.get_atom_coords("N9")
        else:  # Pyrimidines
            n_coords = res.get_atom_coords("N1")
        return np.array(n_coords)

    def get_c1p(res: Residue) -> np.ndarray:
        c1p = res.get_atom_coords("C1'")
        if c1p is None:
            c1p = res.get_atom_coords("C1*")
        return np.array(c1p)

    # Get key atoms for both residues
    n1 = get_glyco_N(res1)
    n2 = get_glyco_N(res2)
    c1p_1 = get_c1p(res1)
    c1p_2 = get_c1p(res2)

    if any(coord is None for coord in [n1, n2, c1p_1, c1p_2]):
        raise ValueError("Missing required atoms for base pair frame construction")

    # Origin at midpoint between glycosidic nitrogens
    origin = (n1 + n2) / 2

    # Y-axis along pseudo-dyad axis (perpendicular to base pair plane)
    # For this we use the cross product of two vectors:
    # 1. Vector between glycosidic nitrogens
    # 2. Vector between one C1' and its glycosidic nitrogen
    v1 = n2 - n1  # Vector between glycosidic nitrogens
    v2 = n1 - c1p_1  # Vector from C1' to N in first residue

    # X-axis along the vector between glycosidic nitrogens
    x_axis = v1 / np.linalg.norm(v1)

    # Y-axis from cross product (perpendicular to base pair plane)
    y_axis = np.cross(v2, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Z-axis completes right-handed system
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

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


def align_basepair_to_identity(
    res1: Residue, res2: Residue
) -> Tuple[np.ndarray, Residue, Residue]:
    """
    Align a base pair to the identity matrix at origin.

    Args:
        res1: First residue of the base pair
        res2: Second residue of the base pair

    Returns:
        Tuple of:
            - 4x4 transformation matrix that aligns base pair to identity
            - Transformed first residue
            - Transformed second residue
    """
    # Get current base pair frame
    current_frame = create_basepair_frame(res1, res2)

    # Get transformation to identity (inverse of current frame)
    transform = np.linalg.inv(current_frame)

    transformed_res1 = get_transformed_residue(res1, transform)
    transformed_res2 = get_transformed_residue(res2, transform)

    return transform, transformed_res1, transformed_res2


def angle_between_base_planes(transform1: np.ndarray, transform2: np.ndarray) -> float:
    """
    Calculate angle between base planes using consistent Z-axis normal vectors.

    Args:
        transform1: 4x4 transformation matrix for first nucleotide
        transform2: 4x4 transformation matrix for second nucleotide
    Returns:
        angle: angle between planes in degrees (0-180)
    """
    # Z-axis is normal to base plane by construction
    normal1 = transform1[:, 2]
    normal2 = transform2[:, 2]

    # Calculate dot product
    dot_product = np.dot(normal1, normal2)

    # Return angle in degrees
    return 90.0 - np.fabs(np.degrees(np.arccos(dot_product)) - 90.0)


def center_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center coordinates by subtracting centroid.

    Args:
        coords: Nx3 array of coordinates

    Returns:
        Tuple of:
            - Centered coordinates
            - Centroid vector
    """
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    return centered_coords, centroid


def kabsch_algorithm(P: list, Q: list) -> list:
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix
    that aligns matrix P to matrix Q.

    Args:
    P (list): A list of coordinates representing the first structure.
    Q (list): A list of coordinates representing the second structure.

    Returns:
    list: The optimal rotation matrix that aligns P to Q.
    """
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    C = np.dot(np.transpose(P_centered), Q_centered)

    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)

    return U


def superimpose_structures(mobile_coords: list, target_coords: list) -> list:
    """
    Superimpose the mobile structure onto the target structure using the Kabsch algorithm.

    Args:
    mobile_coords (list): A list of coordinates for the mobile structure.
    target_coords (list): A list of coordinates for the target structure.

    Returns:
    list: The rotated mobile coordinates that best align with the target structure.
    """
    rotation_matrix = kabsch_algorithm(mobile_coords, target_coords)
    mobile_center = np.mean(mobile_coords, axis=0)
    target_center = np.mean(target_coords, axis=0)

    mobile_coords_aligned = (
        np.dot(mobile_coords - mobile_center, rotation_matrix) + target_center
    )

    return mobile_coords_aligned


def rmsd(mobile_coords: list, target_coords: list) -> float:
    """
    Calculate the RMSD between two sets of coordinates.
    """
    return np.sqrt(np.mean(np.sum((mobile_coords - target_coords) ** 2, axis=1)))
