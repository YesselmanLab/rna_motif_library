import numpy as np
import os
import copy
from rna_motif_library.tranforms import kabsch_rotation, align_and_calc_rmsd
from rna_motif_library.classes import get_residues_from_json


# testing kabsch rotation #####################################################


def test_kabsch_rotation():
    coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    coords2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation, rmsd = kabsch_rotation(coords1, coords2)
    assert rmsd < 1e-6


def test_kabsch_rotation_w_residues():
    json_path = os.path.join("test", "resources", "1A9N_residues.json")
    residues = get_residues_from_json(json_path)
    res = residues["R.G4"]
    coords = res.coords
    angle = np.pi / 4
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    coords_copy = copy.deepcopy(coords)
    coords_copy = np.dot(coords_copy, rotation_matrix.T)

    rotation, transformed_coords, rmsd = align_and_calc_rmsd(coords, coords_copy)
    print(rmsd)
    # assert rmsd < 1e-6


def debug_rotation_alignment():
    # Create simple test coordinates
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Apply rotation
    angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    coords_rotated = np.dot(coords, rotation_matrix.T)

    # Check intermediate values during alignment
    print("Original coordinates:")
    print(coords)
    print("\nRotated coordinates:")
    print(coords_rotated)

    # Get centroids
    centroid1 = np.mean(coords, axis=0)
    centroid2 = np.mean(coords_rotated, axis=0)
    print("\nCentroid 1:", centroid1)
    print("Centroid 2:", centroid2)

    # Check centered coordinates
    coords1_centered = coords - centroid1
    coords2_centered = coords_rotated - centroid2
    print("\nCentered original:")
    print(coords1_centered)
    print("\nCentered rotated:")
    print(coords2_centered)

    # Calculate covariance matrix
    covar = np.dot(coords1_centered.T, coords2_centered)
    print("\nCovariance matrix:")
    print(covar)

    # Do SVD
    U, S, Vt = np.linalg.svd(covar)
    print("\nSingular values:")
    print(S)

    # Check determinant
    d = np.linalg.det(np.dot(Vt.T, U.T))
    print("\nDeterminant:", d)

    # Calculate optimal rotation
    if d < 0:
        Vt[2, :] *= -1
    optimal_rotation = np.dot(Vt.T, U.T)
    print("\nRecovered rotation matrix:")
    print(optimal_rotation)
    print("\nOriginal rotation matrix:")
    print(rotation_matrix)

    # Calculate final RMSD
    coords1_aligned = np.dot(coords1_centered, optimal_rotation)
    rmsd = np.sqrt(np.mean(np.sum((coords1_aligned - coords2_centered) ** 2, axis=1)))
    print("\nFinal RMSD:", rmsd)

    # Now use align_and_calc_rmsd
    rotation, transformed_coords, rmsd = align_and_calc_rmsd(coords, coords_rotated)
    print("\nRMSD from align_and_calc_rmsd:", rmsd)
