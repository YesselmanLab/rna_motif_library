import pandas as pd
import numpy as np

from rna_motif_library.chain import get_cached_protein_chains, get_cached_chains, Chains
from rna_motif_library.residue import get_cached_residues, Residue
from rna_motif_library.util import canon_res_list, ion_list, get_pdb_ids
from typing import Dict, List, Tuple


def is_amino_acid(res) -> bool:
    """Check if residue is an amino acid based on characteristic atoms.

    Args:
        res: Residue object to check

    Returns:
        bool: True if residue appears to be an amino acid, False otherwise
    """
    required_atoms = {"N", "C"}  # Core peptide backbone atoms
    return all(res.get_atom_coords(atom) is not None for atom in required_atoms)


def is_nucleotide(res) -> bool:
    """Check if residue is a nucleotide based on characteristic atoms.

    Args:
        res: Residue object to check

    Returns:
        bool: True if residue appears to be a nucleotide, False otherwise
    """
    required_atoms = {"O3'", "C4'", "C3'", "C5'"}  # Core nucleotide backbone atoms
    return all(res.get_atom_coords(atom) is not None for atom in required_atoms)


def check_residue_bonds(
    residue: Residue, other_residues: Dict[str, Residue], cutoff: float = 2.0
) -> List[Tuple[str, float]]:
    """Check if any atoms in residue could form bonds with atoms in other residues.

    Args:
        residue: The residue to check for bonds
        other_residues: Dictionary of other residues to check against
        cutoff: Maximum distance in Angstroms for atoms to be considered bonded

    Returns:
        List of tuples containing (residue_id, min_distance) for residues with potential bonds
    """
    bonds = []
    res_com = residue.get_center_of_mass()
    res_str = residue.get_str()

    # First filter residues by center of mass distance
    for other_id, other_res in other_residues.items():
        if other_id == res_str:
            continue
        other_com = other_res.get_center_of_mass()
        com_dist = np.linalg.norm(res_com - other_com)

        # Only check atom distances if centers of mass are within 10A
        if com_dist > 15.0:
            continue
        min_dist = float("inf")

        # Check all atom pairs, ignoring hydrogens
        for atom1, coord1 in zip(residue.atom_names, residue.coords):
            if atom1.startswith("H"):
                continue
            for atom2, coord2 in zip(other_res.atom_names, other_res.coords):
                if atom2.startswith("H"):
                    continue
                dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
                min_dist = min(dist, min_dist)

                # Can break early if we find a bond
                if dist < cutoff:
                    bonds.append((other_id, min_dist))
                    break

            if min_dist < cutoff:
                break

    return bonds


def check_ligand_bonds():
    df = pd.read_csv("ligand_instances.csv")
    df["bonded_residues"] = [[] for _ in range(len(df))]
    df = df.query("type == 'SMALL-MOLECULE'")
    exclude = ion_list + ["HOH"]
    for i, g in df.groupby("pdb_id"):
        print(i, len(g))
        residues = get_cached_residues(i)
        keep_residues = {}
        for id, res in residues.items():
            if res.res_id in exclude:
                continue
            keep_residues[id] = res
        for index, row in g.iterrows():
            res = residues[row["res_str"]]
            bonds = check_residue_bonds(res, keep_residues)
            df.at[index, "bonded_residues"] = bonds
    df.to_json("ligand_instances_w_bonds.json", orient="records")
    exit()


def generate_ligand_instances():
    df = pd.read_json("ligand_info_filtered.json")
    df_solvent = pd.read_csv("solvent_and_buffers.csv")
    df_likely_polymer = pd.read_csv("likely_polymer.csv")
    df_non_redundant = pd.read_csv("data/csvs/non_redundant_set.csv")
    exclude = (
        canon_res_list
        + ion_list
        + ["HOH"]
        + ["UNK", "UNX", "N", "DN"]  # unknown residues
        + df_solvent["id"].to_list()
        + df_likely_polymer["id"].to_list()
    )
    # pdb_ids = df_non_redundant["pdb_id"].to_list()
    pdb_ids = get_pdb_ids()
    data = []
    for pdb_id in pdb_ids:
        pchains = Chains(get_cached_protein_chains(pdb_id))
        rchains = Chains(get_cached_chains(pdb_id))
        residues = get_cached_residues(pdb_id)
        for res in residues.values():
            if res.res_id in exclude:
                continue
            is_nuc = is_nucleotide(res)
            is_aa = is_amino_acid(res)
            if is_nuc:
                c = rchains.get_chain_for_residue(res)
                if c is None:
                    continue
                if len(c) > 1:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "RNA",
                            is_nuc,
                            is_aa,
                        ]
                    )
                else:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "SMALL-MOLECULE",
                            is_nuc,
                            is_aa,
                        ]
                    )
            elif is_aa:
                c = pchains.get_chain_for_residue(res)
                if len(c) > 1:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "PROTEIN",
                            is_nuc,
                            is_aa,
                        ]
                    )
                else:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "SMALL-MOLECULE",
                            is_nuc,
                            is_aa,
                        ]
                    )
            else:
                data.append(
                    [
                        pdb_id,
                        res.res_id,
                        res.get_str(),
                        "SMALL-MOLECULE",
                        is_nuc,
                        is_aa,
                    ]
                )
    df = pd.DataFrame(
        data, columns=["pdb_id", "res_id", "res_str", "type", "is_nuc", "is_aa"]
    )
    df.to_csv("ligand_instances.csv", index=False)


def main():
    # check_ligand_bonds()
    df = pd.read_json("ligand_instances_w_bonds.json")
    df = df.query("type == 'SMALL-MOLECULE'")
    df_lig = pd.read_json("ligand_info.json")
    unique_mols = df["res_id"].unique()
    print(len(unique_mols))
    exit()
    for i, row in df.iterrows():
        print(row)
        exit()


if __name__ == "__main__":
    main()
