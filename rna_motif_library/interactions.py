import os
from typing import List, Tuple, Dict
import json
import pandas as pd
import numpy as np

from pydssr.dssr_classes import DSSR_HBOND, DSSR_PAIR
from pydssr.dssr import DSSROutput

from rna_motif_library.classes import (
    X3DNAInteraction,
    Hbond,
    Basepair,
    BasepairParameters,
    Residue,
    X3DNAResidue,
    X3DNAResidueFactory,
    X3DNAPair,
    sanitize_x3dna_atom_name,
)
from rna_motif_library.hbond import HbondFactory, score_hbond
from rna_motif_library.logger import get_logger
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.util import (
    get_nucleotide_atom_type,
    canon_amino_acid_list,
    get_cif_header_str,
)

log = get_logger("interactions")


# top level function ###################################################################


def get_hbonds_from_json(json_path: str) -> List[Hbond]:
    with open(json_path) as f:
        hbonds_data = json.load(f)
        hbonds = [Hbond.from_dict(h) for h in hbonds_data]
    return hbonds


def get_basepairs_from_json(json_path: str) -> List[Basepair]:
    with open(json_path) as f:
        basepairs_data = json.load(f)
        basepairs = [Basepair.from_dict(bp) for bp in basepairs_data]
    return basepairs


def get_hbonds_and_basepairs(
    pdb_name: str, overwrite: bool = False
) -> Tuple[List[Hbond], List[Basepair]]:
    # Check if hbonds and basepairs json files already exist
    """hbonds_json_path = os.path.join(DATA_PATH, "jsons", "hbonds", f"{pdb_name}.json")
    basepairs_json_path = os.path.join(
        DATA_PATH, "jsons", "basepairs", f"{pdb_name}.json"
    )

    if (
        os.path.exists(hbonds_json_path)
        and os.path.exists(basepairs_json_path)
        and not overwrite
    ):
        log.info(f"Loading existing hbonds and basepairs for {pdb_name}")
        hbonds = get_hbonds_from_json(hbonds_json_path)
        basepairs = get_basepairs_from_json(basepairs_json_path)
        return hbonds, basepairs
    log.info(f"Generating hbonds and basepairs for {pdb_name}")"""
    json_path = os.path.join(DATA_PATH, "dssr_output", f"{pdb_name}.json")
    residue_data = json.loads(
        open(os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_name}.json")).read()
    )
    residues = {k: Residue.from_dict(v) for k, v in residue_data.items()}
    dssr_output = DSSROutput(json_path=json_path)
    hbonds = dssr_output.get_hbonds()
    pairs = dssr_output.get_pairs()
    basepairs = get_basepairs(pdb_name, pairs, residues)
    return [], basepairs
    exit()
    save_basepair_params(pairs, pdb_name)
    x3dna_interactions = get_x3dna_interactions(pdb_name, hbonds)
    x3dna_pairs = get_x3dna_pairs(pairs, x3dna_interactions)
    hbonds = get_hbonds(pdb_name, residues, x3dna_interactions, x3dna_pairs)
    basepairs = get_pairs(pdb_name, hbonds, x3dna_pairs)
    return hbonds, basepairs


# handles converting X3DNA to current objects ##########################################


def get_bp_type(bp: str) -> str:
    e = bp.split("-")
    if len(e) != 2:
        e = bp.split("+")
    if len(e) != 2:
        e = [bp[0], bp[-1]]
    if e[0] > e[1]:
        return e[1] + e[0]
    else:
        return e[0] + e[1]


def basepair_to_cif(res1: Residue, res2: Residue, path: str):
    f = open(path, "w")
    f.write(get_cif_header_str())
    acount = 1
    for res in [res1, res2]:
        res_str, acount = res.to_cif_str(acount)
        f.write(res_str)
    f.close()


def get_basepair_info(
    pair: DSSR_PAIR,
    pdb_name: str,
    hbond_score: float,
):
    data = {
        "res_1": pair.nt1.nt_id,
        "res_2": pair.nt2.nt_id,
        "bp_type": get_bp_type(pair.bp),
        "bp_name": pair.name,
        "lw": pair.LW,
        "shear": pair.bp_params[0],
        "stretch": pair.bp_params[1],
        "stagger": pair.bp_params[2],
        "buckle": pair.bp_params[3],
        "propeller": pair.bp_params[4],
        "opening": pair.bp_params[5],
        "hbond_score": hbond_score,
        "pdb_name": pdb_name,
    }
    return data


def get_basepairs(
    pdb_name: str, pairs: List[X3DNAPair], residues: Dict[str, Residue]
) -> List[Basepair]:
    hf = HbondFactory()
    basepairs = []
    all_data = []
    df_bp_hbonds = pd.read_csv("rna_motif_library/resources/basepair_hbonds.csv")
    for pair in pairs.values():
        bp_type = get_bp_type(pair.bp)
        df_sub = df_bp_hbonds[df_bp_hbonds["basepair_type"] == f"{bp_type}_{pair.LW}"]
        res_1 = residues[pair.nt1.nt_id]
        res_2 = residues[pair.nt2.nt_id]
        if res_1.res_id != bp_type[0]:
            res_1, res_2 = res_2, res_1
        h_bond_score = 0
        hbonds = []
        for i, row in df_sub.iterrows():
            hbond_atoms = row["hbond"].split("-")
            hbond = hf.get_hbond(res_1, res_2, hbond_atoms[0], hbond_atoms[1], pdb_name)
            h_bond_score += score_hbond(
                hbond.distance, hbond.angle_1, hbond.angle_2, hbond.dihedral_angle
            )
            hbonds.append(hbond)
        if res_1.res_id == res_2.res_id:
            other_h_bond_score = 0
            other_hbonds = []
            for i, row in df_sub.iterrows():
                hbond_atoms = row["hbond"].split("-")
                hbond = hf.get_hbond(
                    res_1, res_2, hbond_atoms[0], hbond_atoms[1], pdb_name
                )
                other_h_bond_score += score_hbond(
                    hbond.distance, hbond.angle_1, hbond.angle_2, hbond.dihedral_angle
                )
                other_hbonds.append(hbond)
            if other_h_bond_score < h_bond_score:
                hbonds = other_hbonds
        bp_params = BasepairParameters(*pair.bp_params)
        bp = Basepair(
            res_1,
            res_2,
            hbonds,
            bp_type,
            pair.LW,
            pdb_name,
            h_bond_score,
            bp_params,
        )
        basepairs.append(bp)

    # write data to json
    df = pd.DataFrame(all_data)
    df.to_json(
        os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_name}.json"),
        orient="records",
    )
    return basepairs


def get_x3dna_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
    rnp_interactions = parse_snap_output(rnp_out_path)
    rna_interactions = convert_x3dna_hbonds_to_interactions(hbonds)
    all_interactions = merge_hbond_interaction_data(rnp_interactions, rna_interactions)
    return all_interactions


def convert_x3dna_hbonds_to_interactions(
    hbonds: List[DSSR_HBOND],
) -> List[X3DNAInteraction]:
    # Determine type based on atom name

    rna_interactions = []
    for hbond in hbonds:
        atom_1, res_1 = hbond.atom1_id.split("@")
        atom_2, res_2 = hbond.atom2_id.split("@")
        atom_1 = sanitize_x3dna_atom_name(atom_1)
        atom_2 = sanitize_x3dna_atom_name(atom_2)
        type_1 = get_nucleotide_atom_type(atom_1)
        type_2 = get_nucleotide_atom_type(atom_2)
        for aa in canon_amino_acid_list:
            if aa in res_1:
                type_1 = "aa"
            if aa in res_2:
                type_2 = "aa"

        if type_1 == "aa" and type_2 == "aa":
            continue
        # Swap if res_1 is larger than res_2
        if res_1 > res_2:
            atom_1, atom_2 = atom_2, atom_1
            res_1, res_2 = res_2, res_1
            type_1, type_2 = type_2, type_1
        res_1 = X3DNAResidueFactory.create_from_string(res_1)
        res_2 = X3DNAResidueFactory.create_from_string(res_2)
        rna_interactions.append(
            X3DNAInteraction(
                atom_1, res_1, atom_2, res_2, float(hbond.distance), type_1, type_2
            )
        )
    return rna_interactions


def merge_hbond_interaction_data(
    rna_interactions: List[X3DNAInteraction], rnp_interactions: List[X3DNAInteraction]
) -> List[X3DNAInteraction]:
    """
    Merges RNA-RNA and RNA-protein interaction data and removes duplicates and protein-protein interactions.

    Args:
        rna_interactions (list): List of X3DNAInteraction objects from RNA-RNA interactions
        rnp_interactions (list): List of X3DNAInteraction objects from RNA-protein interactions

    Returns:
        unique_interactions (list): List of unique X3DNAInteraction objects, excluding protein-protein interactions
    """
    # Combine all interactions
    all_interactions = rna_interactions + rnp_interactions

    # Create a list of interactions, excluding protein-protein interactions
    unique_interactions = set()
    for interaction in all_interactions:
        # Skip if distance is greater than 3.3 this is not a real hbond
        if interaction.distance > 3.3:
            continue
        # Only keep interactions that aren't between two proteins
        is_protein_protein = (
            interaction.atom_type_1 == "aa" and interaction.atom_type_2 == "aa"
        )
        if not is_protein_protein:
            unique_interactions.add(interaction)

    return list(unique_interactions)


def get_x3dna_pairs(pairs: Dict[str, DSSR_PAIR], interactions: List[X3DNAInteraction]):
    """Get X3DNA pairs from DSSR pairs"""
    x3dna_pairs = []
    for pair in pairs.values():
        res_1 = X3DNAResidueFactory.create_from_string(pair.nt1.nt_id)
        res_2 = X3DNAResidueFactory.create_from_string(pair.nt2.nt_id)
        # Parse hbond description into atom pairs and distances
        pair_interactions = []
        for interaction in interactions:
            if interaction.res_1 == res_1 and interaction.res_2 == res_2:
                pair_interactions.append(interaction)
            elif interaction.res_1 == res_2 and interaction.res_2 == res_1:
                pair_interactions.append(interaction)
        x3dna_pairs.append(
            X3DNAPair(res_1, res_2, pair_interactions, pair.name, pair.LW)
        )
    return x3dna_pairs


# handles converting to final objects ##################################################


def get_pairs(
    pdb_name: str,
    hbonds: List[Hbond],
    x3dna_pairs: List[X3DNAPair],
) -> List[Basepair]:
    pairs = []
    for pair in x3dna_pairs:
        pair_hbonds = []
        for interaction in pair.interactions:
            for hbond in hbonds:
                if (
                    hbond.res_1 == interaction.res_1
                    and hbond.res_2 == interaction.res_2
                    and hbond.atom_1 == interaction.atom_1
                    and hbond.atom_2 == interaction.atom_2
                ):
                    pair_hbonds.append(hbond)
        if len(pair_hbonds) != len(pair.interactions):
            print("Mismatch in number of hbonds and interactions")
            print(len(pair_hbonds), len(pair.interactions))
        pairs.append(
            Basepair(
                pair.res_1,
                pair.res_2,
                pair_hbonds,
                pair.bp_type,
                pair.bp_name,
                pdb_name,
            )
        )
    return pairs
