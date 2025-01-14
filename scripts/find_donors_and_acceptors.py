from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import os
import glob
import wget

from rna_motif_library.residue import (
    Residue,
    get_cached_residues,
    get_residues_from_pdb,
    get_residues_from_cif,
)
from rna_motif_library.chain import write_chain_to_cif
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import canon_res_list, get_pdb_ids, CifParser
from rna_motif_library.x3dna import X3DNAResidue, get_residue_type
from rna_motif_library.residue import sanitize_x3dna_atom_name, Residue


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
    pdb_codes = get_pdb_ids()
    seen = []
    os.makedirs(os.path.join(DATA_PATH, "residues_w_h_pdbs"), exist_ok=True)
    for pdb_code in pdb_codes:
        residues = get_cached_residues(pdb_code)
        for residue in residues.values():
            if residue.res_id in seen:
                continue
            seen.append(residue.res_id)
            org_resid = residue.res_id
            if residue.res_id != "A1H4F":
                continue
            cm = residue.get_center_of_mass()
            residue.move(-cm)
            residue.chain_id = "X"
            residue.num = 1
            residue.ins_code = ""

            # if os.path.exists(
            #     os.path.join(DATA_PATH, "residues_w_h_pdbs", f"{residue.res_id}.pdb")
            # ):
            #     continue
            s = residue.to_pdb_str()
            with open("test.pdb", "w") as f:
                f.write(s)
            os.system(
                f"reduce -BUILD test.pdb -DUMPATOMS test.txt > data/residues_w_h_pdbs/{org_resid}.pdb"
            )
            exit()


def get_residues_with_hydrogen():
    pdb_codes = get_pdb_ids()
    seen = []
    os.makedirs(os.path.join(DATA_PATH, "residues_w_h_cifs"), exist_ok=True)
    for pdb_code in pdb_codes:
        residues = get_cached_residues(pdb_code)
        for residue in residues.values():
            if residue.res_id in seen:
                continue
            seen.append(residue.res_id)
            # Download CIF file for residue type
            cif_url = f"https://files.rcsb.org/ligands/download/{residue.res_id}.cif"
            cif_path = os.path.join(
                DATA_PATH, "residues_w_h_cifs", f"{residue.res_id}.cif"
            )
            print(cif_path)
            if not os.path.exists(cif_path):
                os.system(f"wget {cif_url} -O {cif_path}")


def parse_bad_reduce_pdb(pdb_path: str):
    f = open(pdb_path, "r")
    lines = f.readlines()
    atoms = []
    for line in lines:
        if line.startswith("ATOM"):
            atoms.append(line.split())
    cols = [
        "pdb_group",
        "atom_number",
        "atom_name",
        "residue_name",
        "chain_id",
        "residue_number",
        "x_coord",
        "y_coord",
        "z_coord",
        "occupancy",
        "charge",
        "element",
    ]
    for atom in atoms:
        print(atom)
    df_atom = pd.DataFrame(atoms, columns=cols)
    df_atom["insertion"] = ""
    residues = {}
    for i, g in df_atom.groupby(
        ["chain_id", "residue_number", "residue_name", "insertion"]
    ):
        coords = g[["x_coord", "y_coord", "z_coord"]].values
        atom_names = g["atom_name"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        chain_id, res_num, res_name, ins_code = i
        x3dna_res = X3DNAResidue(
            chain_id, res_name, res_num, ins_code, get_residue_type(res_name)
        )
        residues[x3dna_res.get_str()] = Residue.from_x3dna_residue(
            x3dna_res, atom_names, coords
        )
    return residues


def convert_pdbs_to_cif():
    pdbs = glob.glob(os.path.join(DATA_PATH, "residues_w_h_pdbs", "*.pdb"))
    for pdb in pdbs:
        base_name = os.path.basename(pdb)[:-4]

        # if os.path.exists(
        #     os.path.join(DATA_PATH, "residues_w_h_cifs", f"{base_name}.cif")
        # ):
        #     continue
        h_residues = get_residues_from_pdb(pdb)
        if len(h_residues.values()) != 1:
            print(base_name)
            residues = parse_bad_reduce_pdb(pdb)
        residues = list(h_residues.values())
        write_chain_to_cif(
            residues, os.path.join(DATA_PATH, "residues_w_h_cifs", f"{base_name}.cif")
        )


def get_residue_from_h_cif(cif_path: str):
    print(cif_path)
    parser = CifParser()
    df_atoms = parser.parse(cif_path)
    res_id = df_atoms.iloc[0]["comp_id"]
    atom_names = [
        sanitize_x3dna_atom_name(name) for name in df_atoms["atom_id"].tolist()
    ]
    df_atoms[
        [
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ]
    ] = df_atoms[
        [
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ]
    ].astype(
        float
    )
    coords = df_atoms[
        [
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ]
    ].values

    return Residue("A", res_id, 1, "", "N/A", atom_names, coords)


def find_empty_files(directory: str) -> List[str]:
    """
    Find all empty files in the given directory.

    Args:
        directory (str): Path to directory to search

    Returns:
        List[str]: List of paths to empty files
    """
    empty_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
            empty_files.append(filepath)
    return empty_files


def main():
    residue = get_cached_residues("5HAB")
    unique_res = []
    for res in residue.values():
        if res.res_id not in unique_res:
            unique_res.append(res.res_id)
    print(unique_res)
    exit()
    empty_files = find_empty_files(os.path.join(DATA_PATH, "residues_w_h_cifs"))
    empty_res = [os.path.basename(file)[:-4] for file in empty_files]
    print(empty_res)
    pdb_codes = get_pdb_ids()
    for pdb_code in pdb_codes:
        residues = get_cached_residues(pdb_code)
        for res in residues.values():
            if res.res_id in empty_res:
                print(res.res_id, pdb_code)
                exit()
    exit()

    # generate_residues_with_hydrogen()
    seen = []
    acceptor_donor_data = []
    cif_files = glob.glob(os.path.join(DATA_PATH, "residues_w_h_cifs", "*.cif"))
    for cif_file in cif_files:
        base_name = os.path.basename(cif_file)
        try:
            h_residue = get_residue_from_h_cif(cif_file)
        except Exception as e:
            print(base_name)
            continue
        donors, acceptors = identify_potential_sites(h_residue)
        acceptor_donor_data.append(
            {
                "residue_id": base_name,
                "donors": donors,
                "acceptors": acceptors,
            }
        )

    df = pd.DataFrame(acceptor_donor_data)
    df.to_json("hbond_acceptor_and_donors.json", orient="records")


if __name__ == "__main__":
    main()
