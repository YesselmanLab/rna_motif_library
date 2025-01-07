import os
import copy
import numpy as np
import glob

from biopandas.pdb import PandasPdb

from rna_motif_library.classes import (
    sanitize_x3dna_atom_name,
    get_residues_from_json,
    get_x3dna_res_id,
    X3DNAResidueFactory,
    Residue,
)

from rna_motif_library.interactions import (
    get_basepairs_from_json,
)

from rna_motif_library.tranforms import (
    align_basepair_to_identity,
    NucleotideReferenceFrameGenerator,
)
from rna_motif_library.util import get_cif_header_str


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


def load_ideal_basepairs():
    pdbs = glob.glob(
        os.path.join("rna_motif_library", "resources", "ideal_basepairs", "*.pdb")
    )
    basepairs = {}
    for pdb in pdbs:
        ppdb = PandasPdb().read_pdb(pdb)
        df_atom = ppdb.df["ATOM"]
        residues = []
        for i, g in df_atom.groupby(
            ["chain_id", "residue_number", "residue_name", "insertion"]
        ):
            coords = g[["x_coord", "y_coord", "z_coord"]].values
            atom_names = g["atom_name"].tolist()
            atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
            chain_id, res_num, res_name, ins_code = i
            x3dna_res_id = get_x3dna_res_id(res_name, res_num, chain_id, ins_code)
            x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
            residues.append(Residue.from_x3dna_residue(x3dna_res, atom_names, coords))
        residues = sorted(residues, key=lambda x: x.res_id)
        bp_name = f"{residues[0].res_id}{residues[1].res_id}"
        basepairs[bp_name] = residues
    return basepairs


def basepair_to_cif(res1: Residue, res2: Residue, path: str):
    _, res1, res2 = align_basepair_to_identity(res1, res2)
    f = open(path, "w")
    f.write(get_cif_header_str())
    acount = 1
    for res in [res1, res2]:
        res_str, acount = res.to_cif_str(acount)
        f.write(res_str)
    f.close()


def main():
    ideal_basepairs = load_ideal_basepairs()
    json_path = os.path.join("data", "jsons", "basepairs", "7MQA.json")
    basepairs = get_basepairs_from_json(json_path)
    json_path = os.path.join("data", "jsons", "residues", "7MQA.json")
    residues = get_residues_from_json(json_path)
    for bp in basepairs:
        if bp.bp_name != "cWW":
            continue
        bp_id = bp.res_1.res_id + bp.res_2.res_id
        if bp_id not in ideal_basepairs:
            continue
        residues_ideal = ideal_basepairs[bp_id]
        residues_motif = []
        for res_id in [bp.res_1.get_str(), bp.res_2.get_str()]:
            residues_motif.append(residues[res_id])
        ideal_coords = []
        motif_coords = []
        for res in residues_ideal:
            ideal_coords.extend(res.coords)
        for res in residues_motif:
            motif_coords.extend(res.coords)
        ideal_coords = np.array(ideal_coords)
        motif_coords = np.array(motif_coords)
        ideal_coords = superimpose_structures(ideal_coords, motif_coords)
        print(rmsd(ideal_coords, motif_coords))
        basepair_to_cif(residues_motif[0], residues_motif[1], f"motif_{bp.bp_name}.cif")
        basepair_to_cif(residues_ideal[0], residues_ideal[1], f"ideal_{bp.bp_name}.cif")
        exit()


if __name__ == "__main__":
    main()
