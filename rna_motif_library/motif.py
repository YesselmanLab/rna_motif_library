# Standard library imports
import json
import os
import glob
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# Third party imports
import numpy as np
import pandas as pd
import click
from pydssr.dssr import DSSROutput

# Local imports
from rna_motif_library.basepair import Basepair, get_cached_basepairs
from rna_motif_library.chain import Chains, get_rna_chains, get_cached_chains
from rna_motif_library.hbond import Hbond, get_cached_hbonds
from rna_motif_library.logger import get_logger
from rna_motif_library.residue import Residue, get_cached_residues
from rna_motif_library.settings import RESOURCES_PATH, DATA_PATH
from rna_motif_library.util import (
    get_cif_header_str,
    get_cached_path,
    add_motif_indentifier_columns,
    get_pdb_ids,
    NRSEntry,
    file_exists_and_has_content,
    parse_motif_indentifier,
    NonRedundantSetParser,
)
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)
from rna_motif_library.x3dna import X3DNAResidue
from rna_motif_library.tranforms import superimpose_structures, rmsd
from rna_motif_library.pdb_data import (
    PDBStructureData,
    get_pdb_structure_data,
    get_cww_basepairs,
    get_singlet_pairs,
    get_basepair_ends_for_strands,
)

log = get_logger("motif")


def get_motifs_from_dssr(pdb_id: str) -> list:
    residues = get_cached_residues(pdb_id)
    basepairs = get_cached_basepairs(pdb_id)
    hbonds = get_cached_hbonds(pdb_id)
    chains = Chains(get_rna_chains(residues.values()))
    mf = MotifFactoryFromOther(pdb_id, chains, hbonds, basepairs)
    return mf.get_motifs_from_dssr()


class Motif:
    def __init__(
        self,
        name: str,
        mtype: str,
        pdb: str,
        size: str,
        sequence: str,
        strands: List[List[Residue]],
        basepairs: List[Basepair],
        basepair_ends: List[Tuple[str, str]],
        hbonds: List[Hbond],
        quality_score: float = -1.0,
    ):
        self.name = name
        self.mtype = mtype
        self.pdb = pdb
        self.size = size
        self.sequence = sequence
        self.strands = strands
        self.basepairs = basepairs
        self.basepair_ends = basepair_ends
        self.hbonds = hbonds
        self.quality_score = quality_score

    @classmethod
    def from_dict(cls, d: dict):
        # Convert nested objects back to their proper classes
        strands = []
        for strand in d["strands"]:
            strands.append([Residue.from_dict(r) for r in strand])
        basepairs = [Basepair.from_dict(bp) for bp in d["basepairs"]]
        basepair_ends = [Basepair.from_dict(bp) for bp in d["basepair_ends"]]
        hbonds = [Hbond.from_dict(hb) for hb in d["hbonds"]]
        if "quality_score" in d:
            quality_score = d["quality_score"]
        else:
            quality_score = -1.0

        return cls(
            name=d["name"],
            mtype=d["mtype"],
            pdb=d["pdb"],
            size=d["size"],
            sequence=d["sequence"],
            strands=strands,
            basepairs=basepairs,
            basepair_ends=basepair_ends,
            hbonds=hbonds,
            quality_score=quality_score,
        )

    def is_equal(self, other, check_coords=False):
        is_equal = (
            (self.name == other.name)
            and (self.mtype == other.mtype)
            and (self.sequence == other.sequence)
            and (self.size == other.size)
        )
        if check_coords:
            for strand1, strand2 in zip(self.strands, other.strands):
                for res1, res2 in zip(strand1, strand2):
                    if res1.is_equal(res2, check_coords):
                        continue
                    else:
                        return False
        return is_equal

    def num_strands(self):
        return len(self.strands)

    def num_residues(self):
        return sum(len(strand) for strand in self.strands)

    def num_basepairs(self):
        return len(self.basepairs)

    def num_hbonds(self):
        return len(self.hbonds)

    def num_basepair_ends(self):
        return len(self.basepair_ends)

    def get_residues(self):
        return [res for strand in self.strands for res in strand]

    def get_phos_coords(self):
        coords = []
        for res in self.get_residues():
            if res.get_atom_coords("P") is None:
                continue
            coords.append(res.get_atom_coords("P"))
        return np.array(coords)

    def get_c1prime_coords(self):
        coords = []
        for strand in self.strands:
            for res in strand:
                if res.get_atom_coords("C1'") is None:
                    continue
                coords.append(res.get_atom_coords("C1'"))
        return np.array(coords)

    def contains_residue(self, residue: Residue) -> bool:
        for strand in self.strands:
            for res in strand:
                if res.get_str() == residue.get_str():
                    return True
        return False

    def contains_basepair(self, basepair: Basepair) -> bool:
        if basepair in self.basepairs:
            return True
        return False

    def contains_hbond(self, hbond: Hbond) -> bool:
        if hbond in self.hbonds:
            return True
        return False

    def get_basepair(self, res1: Residue, res2: Residue) -> Basepair:
        for bp in self.basepairs:
            if (
                bp.res_1.get_str() == res1.get_str()
                and bp.res_2.get_str() == res2.get_str()
            ):
                return bp
            if (
                bp.res_1.get_str() == res2.get_str()
                and bp.res_2.get_str() == res1.get_str()
            ):
                return bp
        return None

    def get_basepair_for_residue(self, res: Residue) -> Basepair:
        for bp in self.basepairs:
            if (
                bp.res_1.get_str() == res.get_str()
                or bp.res_2.get_str() == res.get_str()
            ):
                return bp
        return None

    def residue_has_basepair(self, res: Residue) -> bool:
        for bp in self.basepairs:
            if (
                bp.res_1.get_str() == res.get_str()
                or bp.res_2.get_str() == res.get_str()
            ):
                return True
        return False

    def to_dict(self):
        strands = []
        for strand in self.strands:
            strands.append([res.to_dict() for res in strand])
        return {
            "name": self.name,
            "mtype": self.mtype,
            "pdb": self.pdb,
            "size": self.size,
            "sequence": self.sequence,
            "strands": strands,
            "basepairs": [bp.to_dict() for bp in self.basepairs],
            "basepair_ends": [bp.to_dict() for bp in self.basepair_ends],
            "hbonds": [hb.to_dict() for hb in self.hbonds],
        }

    def to_cif(self, cif_path: Optional[str] = None):
        if cif_path is None:
            cif_path = f"{self.name}.cif"
        f = open(cif_path, "w")
        f.write(get_cif_header_str())
        acount = 1
        f.write(self.to_cif_str(acount))
        f.close()

    def to_cif_str(self, acount: int):
        s = ""
        for residue in self.get_residues():
            s += residue.to_cif_str(acount)
            acount += len(residue.atom_names)
        return s


def get_motifs_from_json(json_path: str) -> List[Motif]:
    """
    Get motifs from a JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return [Motif.from_dict(d) for d in data]


def save_motifs_to_json(motifs: List[Motif], json_path: str):
    """Save motifs to a JSON file"""
    with open(json_path, "w") as f:
        json.dump([m.to_dict() for m in motifs], f)


def get_cached_motifs(pdb_id: str) -> List[Motif]:
    json_path = get_cached_path(pdb_id, "motifs")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Motifs file not found for {pdb_id}")
    return get_motifs_from_json(json_path)
