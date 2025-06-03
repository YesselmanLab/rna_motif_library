import glob
import os
import numpy as np


from rna_motif_library.classes import (
    sanitize_x3dna_atom_name,
    get_residues_from_json,
    get_x3dna_res_id,
    X3DNAResidueFactory,
    Residue,
)

from rna_motif_library.resources import load_ideal_bases, load_ideal_basepairs
from rna_motif_library.util import purine_atom_names, pyrimidine_atom_names


class NucleotideReferenceFrameGenerator:
    def __init__(self):
        self.ideal_bases = load_ideal_bases()
        self.ideal_basepairs = load_ideal_basepairs()

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
        if residue.res_id in ["A", "G"]:
            base_atoms = purine_atom_names
        elif residue.res_id in ["U", "C"]:
            base_atoms = pyrimidine_atom_names
        else:
            raise ValueError(f"Unknown residue: {residue.res_name}")
        coords = []
        for atom in base_atoms:
            coords.append(residue.coords[residue.atom_names.index(atom)])
        return np.array(coords)


# Example usage:
if __name__ == "__main__":
    basepairs = load_ideal_basepairs()
    fg = NucleotideReferenceFrameGenerator()
    frame = fg.get_reference_frame(basepairs["AU"][0])
    print(frame)
