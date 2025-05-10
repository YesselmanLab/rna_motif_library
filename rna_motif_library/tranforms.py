import os
import copy
import json
import numpy as np
import pandas as pd

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.resources import Residue
from rna_motif_library.resources import Residue
from rna_motif_library.util import purine_atom_names, pyrimidine_atom_names
import numpy as np
from typing import Tuple, List, Optional, Dict


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


def align_motif(mobile_motif, target_motif):
    coords_1 = mobile_motif.get_c1prime_coords()
    coords_2 = target_motif.get_c1prime_coords()
    if len(coords_1) != len(coords_2):
        return None
    rotation_matrix = kabsch_algorithm(coords_1, coords_2)
    new_m = copy.deepcopy(mobile_motif)
    mobile_center = np.mean(coords_1, axis=0)
    target_center = np.mean(coords_2, axis=0)
    for strand in new_m.strands:
        for res in strand:
            res.coords = (
                np.dot(res.coords - mobile_center, rotation_matrix) + target_center
            )
    return new_m


def pymol_align(mobile_coords: np.ndarray, target_coords: np.ndarray, max_cycles: int = 5, cutoff_factor: float = 2.0) -> Tuple[np.ndarray, float, Dict]:
    """
    Implements PyMOL-style iterative alignment between two sets of coordinates.
    
    This function mimics PyMOL's alignment algorithm by:
    1. Performing initial alignment of all matched atoms
    2. Iteratively rejecting outliers and refining the alignment
    3. Continuing until convergence or max cycles reached
    
    Args:
        mobile_coords: Coordinates of mobile structure to be aligned (N x 3 array)
        target_coords: Coordinates of target structure (N x 3 array)
        max_cycles: Maximum number of refinement cycles (default: 5)
        cutoff_factor: Factor for outlier rejection (default: 2.0)
        
    Returns:
        Tuple containing:
        - Aligned coordinates of mobile structure
        - Final RMSD value
        - Dictionary with alignment statistics
    """
    if len(mobile_coords) != len(target_coords):
        raise ValueError(f"Number of atoms must match: {len(mobile_coords)} vs {len(target_coords)}")
    
    n_atoms = len(mobile_coords)
    # print(f"MatchAlign: aligning atoms ({n_atoms} vs {n_atoms})...")
    
    # Initial alignment with all atoms
    aligned_coords = superimpose_structures(mobile_coords, target_coords)
    current_rmsd = rmsd(aligned_coords, target_coords)
    # print(f"MatchAlign: score {current_rmsd:.3f}")
    # print(f"ExecutiveAlign: {n_atoms} atoms aligned.")
    
    # Keep track of which atoms to include in alignment
    atom_mask = np.ones(n_atoms, dtype=bool)
    stats = {
        "cycles": [], 
        "rejected_atoms": [], 
        "rmsd_values": [current_rmsd],
        "final_mask": atom_mask  # Store the mask for later use
    }
    
    # Iterative refinement
    for cycle in range(max_cycles):
        # Calculate per-atom distances
        distances = np.sqrt(np.sum((aligned_coords - target_coords) ** 2, axis=1))
        
        # Only consider atoms that are currently included
        active_distances = distances[atom_mask]
        
        if len(active_distances) == 0:
            break
            
        # Calculate distance statistics
        mean_dist = np.mean(active_distances)
        std_dist = np.std(active_distances)
        
        # Set rejection threshold (PyMOL uses a dynamic threshold)
        cutoff = mean_dist + cutoff_factor * std_dist
        
        # Find outliers
        outliers = atom_mask & (distances > cutoff)
        n_rejected = np.sum(outliers)
        
        # If no outliers were found, we're done
        if n_rejected == 0:
            break
            
        # Remove outliers from consideration
        atom_mask[outliers] = False
        n_remaining = np.sum(atom_mask)
        
        # Re-align with remaining atoms
        if n_remaining >= 3:  # Need at least 3 points for alignment
            # Get rotation and translation for alignment
            rotation_matrix = kabsch_algorithm(
                mobile_coords[atom_mask] - np.mean(mobile_coords[atom_mask], axis=0),
                target_coords[atom_mask] - np.mean(target_coords[atom_mask], axis=0)
            )
            mobile_center = np.mean(mobile_coords[atom_mask], axis=0)
            target_center = np.mean(target_coords[atom_mask], axis=0)
            
            # Transform all atoms, not just the ones used in alignment
            aligned_coords = np.dot(mobile_coords - mobile_center, rotation_matrix) + target_center
            
            # Calculate new RMSD on included atoms only
            current_rmsd = rmsd(aligned_coords[atom_mask], target_coords[atom_mask])
            
            # print(f"ExecutiveRMS: {n_rejected} atoms rejected during cycle {cycle+1} (RMSD={current_rmsd:.2f}).")
            
            # Store statistics
            stats["cycles"].append(cycle+1)
            stats["rejected_atoms"].append(n_rejected)
            stats["rmsd_values"].append(current_rmsd)
        else:
            # Not enough atoms left for alignment
            print("Warning: Too few atoms remaining for alignment")
            break
    
    # Update final mask
    stats["final_mask"] = atom_mask
    
    # Final alignment report
    n_aligned = np.sum(atom_mask)
    final_rmsd = rmsd(aligned_coords[atom_mask], target_coords[atom_mask])
    # print(f"Executive: RMSD = {final_rmsd:.3f} ({n_aligned} to {n_aligned} atoms)")
    
    return aligned_coords, final_rmsd, stats