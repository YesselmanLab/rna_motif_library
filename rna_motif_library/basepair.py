import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pydssr.dssr_classes import DSSR_HBOND, DSSR_PAIR

from rna_motif_library.util import *
from rna_motif_library.hbond import Hbond, HbondFactory, score_hbond
from rna_motif_library.x3dna import X3DNAResidue, X3DNAPair
from rna_motif_library.residue import Residue
from rna_motif_library.logger import get_logger

log = get_logger("basepair")


@dataclass(frozen=True, order=True)
class BasepairParameters:
    shear: float
    stretch: float
    stagger: float
    buckle: float
    propeller: float
    opening: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasepairParameters":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass(frozen=True, order=True)
class Basepair:
    res_1: X3DNAResidue
    res_2: X3DNAResidue
    hbonds: List[Hbond]
    bp_type: str
    lw: str
    pdb_name: str
    hbond_score: float
    bp_params: BasepairParameters

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Basepair":
        """
        Creates a Basepair instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing Basepair attributes

        Returns:
            Basepair: New Basepair instance
        """
        # Convert X3DNAInteraction objects
        data["res_1"] = X3DNAResidue.from_dict(data["res_1"])
        data["res_2"] = X3DNAResidue.from_dict(data["res_2"])
        # Convert list of Hbond objects
        data["hbonds"] = [Hbond.from_dict(hbond) for hbond in data["hbonds"]]
        data["bp_params"] = BasepairParameters.from_dict(data["bp_params"])
        return cls(**data)

    def __hash__(self):
        # Create a unique hash based on the residues and score
        return hash((self.res_1.get_str(), self.res_2.get_str(), self.hbond_score))

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts Pair instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing Basepair attributes
        """
        data = vars(self).copy()
        data["res_1"] = self.res_1.to_dict()
        data["res_2"] = self.res_2.to_dict()
        data["hbonds"] = [hbond.to_dict() for hbond in self.hbonds]
        data["bp_params"] = self.bp_params.to_dict()
        return data

    def get_partner(self, x3dna_id: str):
        if self.res_1.get_str() == x3dna_id:
            return self.res_2
        elif self.res_2.get_str() == x3dna_id:
            return self.res_1
        else:
            return None


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


def get_basepairs_from_json(json_path: str) -> List[Basepair]:
    with open(json_path) as f:
        basepairs_data = json.load(f)
        basepairs = [Basepair.from_dict(bp) for bp in basepairs_data]
    return basepairs
