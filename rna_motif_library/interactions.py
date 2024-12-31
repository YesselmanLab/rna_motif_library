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
    get_basepairs_from_json,
)
from rna_motif_library.hbond import HbondFactory, score_hbond, parse_hbond_description
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
    hbonds_json_path = os.path.join(DATA_PATH, "jsons", "hbonds", f"{pdb_name}.json")
    basepairs_json_path = os.path.join(
        DATA_PATH, "jsons", "basepairs", f"{pdb_name}.json"
    )

    if (
        os.path.exists(hbonds_json_path)
        and os.path.exists(basepairs_json_path)
        and not overwrite
    ):
        log.info(f"Loading existing hbonds and basepairs for {pdb_name}")
        # hbonds = get_hbonds_from_json(hbonds_json_path)
        basepairs = get_basepairs_from_json(basepairs_json_path)
        return [], basepairs
    log.info(f"Generating hbonds and basepairs for {pdb_name}")
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
        return e[1] + "-" + e[0]
    else:
        return e[0] + "-" + e[1]


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
        "ref_frame": np.array(
            [
                pair.frame["x_axis"],
                pair.frame["y_axis"],
                pair.frame["z_axis"],
            ]
        ),
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


class BasepairFactory:
    def __init__(self):
        self.df_bp_hbonds = pd.read_csv(
            "rna_motif_library/resources/basepair_hbonds.csv"
        )
        df_acceptor_donors = pd.read_json(
            "rna_motif_library/resources/hbond_acceptor_and_donors.json"
        )
        self.hbond_acceptors = {}
        self.hbond_donors = {}
        for i, row in df_acceptor_donors.iterrows():
            self.hbond_acceptors[row["residue_id"]] = row["acceptors"]
            self.hbond_donors[row["residue_id"]] = row["donors"]
        self.hf = HbondFactory()

    def get_basepair(
        self, pdb_name: str, pair: DSSR_PAIR, residues: Dict[str, Residue]
    ) -> Basepair:
        res_1, res_2 = self._get_bp_residues(pdb_name, pair, residues)
        if res_1 is None or res_2 is None:
            return None
        bp_type = get_bp_type(pair.bp)
        bp_type_short = bp_type.replace("-", "")
        if pair.nt1.nt_id != res_1.res_id:
            res_1, res_2 = res_2, res_1
        df_sub = self.df_bp_hbonds[
            self.df_bp_hbonds["basepair_type"] == f"{bp_type_short}_{pair.LW}"
        ]
        if len(df_sub) == 0:
            hbonds = self._get_potential_hbonds(res_1, res_2, pdb_name)
        for _, row in df_sub.iterrows():
            hbonds = self._get_hbonds_from_known_iteractions(
                res_1, res_2, df_sub, pdb_name
            )
            if res_1.res_id == res_2.res_id:
                other_hbonds = self._get_hbonds_from_known_iteractions(
                    res_2, res_1, df_sub, pdb_name
                )
                if self._get_hbond_score(other_hbonds) < self._get_hbond_score(hbonds):
                    hbonds = other_hbonds
        hbond_score = self._get_hbond_score(hbonds)
        bp_params = BasepairParameters(*pair.bp_params)
        bp = Basepair(
            res_1.get_x3dna_residue(),
            res_2.get_x3dna_residue(),
            hbonds,
            bp_type,
            pair.LW,
            pdb_name,
            hbond_score,
            bp_params,
        )
        return bp

    def _get_bp_residues(
        self, pdb_name: str, pair: X3DNAPair, residues: Dict[str, Residue]
    ):
        try:
            res_1 = residues[pair.nt1.nt_id]
            res_2 = residues[pair.nt2.nt_id]
        except KeyError:
            log.error(
                f"Residue not found in residues: {pdb_name}, {pair.nt1.nt_id}, {pair.nt2.nt_id}"
            )
            return None, None
        return res_1, res_2

    def _get_potential_hbonds(self, res_1: Residue, res_2: Residue, pdb_name: str):
        hbond_atom_pairs = []
        acceptors = self.hbond_acceptors[res_1.res_id]
        donors = self.hbond_donors[res_2.res_id]
        for acceptor in acceptors:
            for donor in donors:
                hbond_atom_pairs.append((acceptor, donor))
        acceptors = self.hbond_acceptors[res_2.res_id]
        donors = self.hbond_donors[res_1.res_id]
        for acceptor in acceptors:
            for donor in donors:
                hbond_atom_pairs.append((acceptor, donor))
        potential_hbonds = []
        for res1_atom, res2_atom in hbond_atom_pairs:
            hbond = self.hf.get_hbond(res_1, res_2, res1_atom, res2_atom, pdb_name)
            if hbond is None:
                continue
            potential_hbonds.append(hbond)
        # Score all potential hbonds
        scored_hbonds = []
        for hbond in potential_hbonds:
            score = score_hbond(
                hbond.distance, hbond.angle_1, hbond.angle_2, hbond.dihedral_angle
            )
            scored_hbonds.append((score, hbond))

        # Sort by score descending
        scored_hbonds.sort(reverse=True, key=lambda x: x[0])

        # Track which atoms have been used
        used_atoms_res1 = set()
        used_atoms_res2 = set()

        # Pick best scoring hbonds where atoms haven't been used
        final_hbonds = []
        for score, hbond in scored_hbonds:
            if (
                hbond.atom_1 not in used_atoms_res1
                and hbond.atom_2 not in used_atoms_res2
            ):
                final_hbonds.append(hbond)
                used_atoms_res1.add(hbond.atom_1)
                used_atoms_res2.add(hbond.atom_2)

        potential_hbonds = final_hbonds
        return potential_hbonds

    def _get_hbonds_from_known_iteractions(
        self, res_1: Residue, res_2: Residue, df_sub: pd.DataFrame, pdb_name: str
    ):
        hbonds = []
        for i, row in df_sub.iterrows():
            hbond_atoms = row["hbond"].split("-")
            hbond = self.hf.get_hbond(
                res_1, res_2, hbond_atoms[0], hbond_atoms[1], pdb_name
            )
            if hbond is None:
                continue
            hbonds.append(hbond)
        return hbonds

    def _get_hbond_score(self, hbonds: List[Hbond]):
        hbond_score = 0
        for hbond in hbonds:
            hbond_score += score_hbond(
                hbond.distance, hbond.angle_1, hbond.angle_2, hbond.dihedral_angle
            )
        return hbond_score


def get_basepairs(
    pdb_name: str, pairs: List[DSSR_PAIR], residues: Dict[str, Residue]
) -> List[Basepair]:
    bf = BasepairFactory()
    basepairs = []
    all_data = []
    for pair in pairs.values():
        basepair = bf.get_basepair(pdb_name, pair, residues)
        if basepair is None:
            log.error(
                f"Basepair not found for {pdb_name}, {pair.nt1.nt_id}, {pair.nt2.nt_id}"
            )
            continue
        basepairs.append(basepair)
        data = get_basepair_info(pair, pdb_name, basepair.hbond_score)
        all_data.append(data)

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
