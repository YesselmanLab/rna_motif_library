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
    add_motif_name_columns,
    get_pdb_ids,
    NRSEntry,
    file_exists_and_has_content,
    parse_motif_name,
    NonRedundantSetParser,
)
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)
from rna_motif_library.x3dna import X3DNAResidue
from rna_motif_library.tranforms import superimpose_structures, rmsd
from rna_motif_library.pdb_data import (
    get_pdb_structure_data,
    get_cww_basepairs,
    get_singlet_pairs,
)

log = get_logger("motif")


def get_motifs(pdb_id: str) -> list:
    residues = get_cached_residues(pdb_id)
    basepairs = get_cached_basepairs(pdb_id)
    hbonds = get_cached_hbonds(pdb_id)
    chains = Chains(get_rna_chains(residues.values()))
    mf = MotifFactory(pdb_id, chains, basepairs, hbonds)
    return mf.get_motifs()


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


# MotifFactory #######################################################################


class MotifFactory:
    def __init__(
        self,
        pdb_id: str,
        chains: Chains,
        basepairs: List[Basepair],
        hbonds: List[Hbond],
    ):
        self.pdb_id = pdb_id
        self.chains = chains
        self.basepairs = basepairs
        self.hbonds = hbonds
        self.motif_name_count = {}
        self.cww_basepairs_lookup = self._get_cww_basepairs(self.basepairs)
        self.cww_basepairs = set(self.cww_basepairs_lookup.values())
        self.cww_residues_to_basepairs = {}
        # Group basepairs by residue
        residue_basepairs = {}
        for bp in self.cww_basepairs:
            if bp.res_1.get_str() not in residue_basepairs:
                residue_basepairs[bp.res_1.get_str()] = []
            if bp.res_2.get_str() not in residue_basepairs:
                residue_basepairs[bp.res_2.get_str()] = []
            residue_basepairs[bp.res_1.get_str()].append(bp)
            residue_basepairs[bp.res_2.get_str()].append(bp)

        # For each residue, take only the basepair with highest hbond score
        for res, bps in residue_basepairs.items():
            if len(bps) != 1:
                log.warn(
                    f"Residue {res} has {len(bps)} basepairs. Only taking the one with the highest hbond score."
                )
                for bp in bps:
                    log.warn(
                        f"Basepair: {bp.res_1.get_str()} {bp.res_2.get_str()} with hbond score {bp.hbond_score}"
                    )
            best_bp = max(bps, key=lambda x: x.hbond_score)
            self.cww_residues_to_basepairs[res] = best_bp
        self.singlet_pairs = self._find_singlet_pairs()
        self.singlet_pairs_lookup = {}
        for bp in self.singlet_pairs:
            self.singlet_pairs_lookup[bp.res_1.get_str() + "-" + bp.res_2.get_str()] = (
                bp
            )
            self.singlet_pairs_lookup[bp.res_2.get_str() + "-" + bp.res_1.get_str()] = (
                bp
            )

    # callable functions ################################################################
    # main interface
    def get_motifs(self) -> List[Motif]:
        possible_hairpins = self.get_looped_strands()
        helices = self.get_helices(possible_hairpins)
        non_helical_strands = self.get_non_helical_strands(helices)
        strands_between_helices = self.get_strands_between_helices(helices)
        missing_residues = self.get_missing_residues(helices, non_helical_strands)
        if len(missing_residues) > 0:
            log.error(f"Missing residues: {[x.get_str() for x in missing_residues]}")
            exit()
        else:
            log.info("No missing residues after helices and non-helical strands")
        potential_motifs = self.get_non_canonical_motifs(
            non_helical_strands + strands_between_helices
        )
        # for i, m in enumerate(potential_motifs):
        #    m.to_cif(f"{m.name}.cif")
        missing_residues = self.get_missing_residues(helices + potential_motifs)
        if len(missing_residues) > 0:
            log.error(f"Missing residues: {[x.get_str() for x in missing_residues]}")
            exit()
        else:
            log.info("No missing residues after helices and non-helical motifs")
        finished_motifs, unfinished_motifs = self._find_and_assign_finished_motifs(
            potential_motifs
        )
        missing_residues = self.get_missing_residues(
            helices + finished_motifs + unfinished_motifs
        )
        if len(missing_residues) > 0:
            log.error(f"Missing residues: {[x.get_str() for x in missing_residues]}")
            exit()
        for m in unfinished_motifs:
            new_motif = self.get_bulge_or_multiway_junction(
                m.strands[0], strands_between_helices
            )
            if new_motif is None:
                finished_motifs.append(m)
                continue
            else:
                finished_motifs.append(new_motif)
        finalized_motifs = []
        self.helical_residues = []
        for m in helices:
            for res in m.get_residues():
                self.helical_residues.append(res.get_str())
        for m in finished_motifs + helices:
            new_m = self._finalize_motif(m)
            if new_m is None:
                continue
            finalized_motifs.append(new_m)
        missing_residues = self.get_missing_residues(finalized_motifs)
        if len(missing_residues) > 0:
            log.error(f"Missing residues: {[x.get_str() for x in missing_residues]}")
            exit()
        return finalized_motifs

    def get_helices(self, hairpins: List[Motif]) -> List[Motif]:
        """Find all possible helices in the structure."""
        # Get residues already used in hairpins
        # this avoids confusing pseudoknots with helices
        hairpin_x3nda_strs = []
        for h in hairpins:
            # exclude first and last residue which are a cWW basepair
            for res in h.strands[0][1:-1]:
                hairpin_x3nda_strs.append(res.get_str())
        # Filter to only valid cWW basepairs not in hairpins
        cww_basepairs = []
        for bp in self.cww_basepairs:
            if (
                bp.res_1.get_str() in hairpin_x3nda_strs
                or bp.res_2.get_str() in hairpin_x3nda_strs
            ):
                continue
            cww_basepairs.append(bp)
        helices = []
        used_basepairs = set()
        # Look for consecutive basepairs
        for bp in cww_basepairs:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in used_basepairs:
                continue
            helix = self._build_helix(bp, hairpin_x3nda_strs, used_basepairs)
            if helix:
                helices.append(helix)

        return helices

    def get_strands_between_helices(self, helices: List[Motif]) -> List[List[Residue]]:
        strands = []
        # Map residues to their helices
        helix_map = {}
        for i, helix in enumerate(helices):
            for res in helix.get_residues():
                helix_map[res.get_str()] = i
        # Look for consecutive residues that span different helices
        for chain in self.chains.chains:
            for i in range(len(chain) - 1):
                res1 = chain[i]
                res2 = chain[i + 1]
                # Skip if either residue isn't in a helix
                if res1.get_str() not in helix_map or res2.get_str() not in helix_map:
                    continue
                # Check if residues are in different helices
                if helix_map[res1.get_str()] != helix_map[res2.get_str()]:
                    strands.append([res1, res2])

        return strands

    def get_non_canonical_motifs(
        self,
        non_helical_strands: List[List[Residue]],
    ) -> List[Motif]:

        # Group strands that share basepairs
        motifs = []
        unprocessed_strands = non_helical_strands.copy()
        count = 0

        while unprocessed_strands:
            # Start a new motif with the first unprocessed strand
            count += 1
            motifs.extend(
                self._get_non_canonical_motif_from_strands(
                    unprocessed_strands[0], count, unprocessed_strands
                )
            )

        return motifs

    def get_missing_residues(
        self,
        motifs: List[Motif],
        strands: List[List[Residue]] = None,
    ) -> List[Residue]:
        """Get residues not included in any motif or strand.

        Args:
            motifs: List of motifs to check
            strands: Optional list of strands to check

        Returns:
            List of residues not in any motif or strand
        """
        motifs_residues = []
        for m in motifs:
            motifs_residues.extend(m.get_residues())

        if strands is not None:
            for strand in strands:
                motifs_residues.extend(strand)

        return [r for r in self.chains.get_residues() if r not in motifs_residues]

    # basepair functions ################################################################

    def _get_cww_basepairs(self, basepairs: List[Basepair]) -> Dict[str, Basepair]:
        """Get dictionary of cWW basepairs keyed by residue pair strings."""
        allowed_pairs = []
        f = open(os.path.join(RESOURCES_PATH, "valid_cww_pairs.txt"))
        lines = f.readlines()
        for line in lines:
            allowed_pairs.append(line.strip())
        f.close()
        cww_basepairs = {}
        two_hbond_pairs = ["A-U", "U-A", "G-U", "U-G"]
        three_hbond_pairs = ["G-C", "C-G"]
        for bp in basepairs:
            if bp.lw != "cWW" or bp.bp_type not in allowed_pairs:
                continue
            if (
                self.chains.get_residue_by_str(bp.res_1.get_str()) is None
                or self.chains.get_residue_by_str(bp.res_2.get_str()) is None
            ):
                continue
            # stops a lot of bad basepairs from being included
            if bp.bp_type in two_hbond_pairs and bp.hbond_score < 1.3:
                continue
            if bp.bp_type in three_hbond_pairs and bp.hbond_score < 2.0:
                continue
            key1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            key2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
            cww_basepairs[key1] = bp
            cww_basepairs[key2] = bp
        return cww_basepairs

    def _get_possible_flanking_bps(self, bp: Basepair) -> List[Basepair]:
        res1, res2 = self.chains.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.chains.get_residue_position(res1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res2)
        res1_next = self.chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
        res1_prev = self.chains.get_residue_by_pos(chain_num_1, pos_1 - 1)
        res2_next = self.chains.get_residue_by_pos(chain_num_2, pos_2 + 1)
        res2_prev = self.chains.get_residue_by_pos(chain_num_2, pos_2 - 1)
        combos = [
            (res1_next, res2_prev),
            (res1_next, res2_next),
            (res1_prev, res2_prev),
            (res1_prev, res2_next),
        ]
        return combos

    def _number_of_flanking_bps(self, bp: Basepair) -> int:
        combos = self._get_possible_flanking_bps(bp)
        count = 0
        for res1, res2 in combos:
            if res1 is None or res2 is None:
                continue
            key = f"{res1.get_str()}-{res2.get_str()}"
            if key in self.cww_basepairs_lookup:
                count += 1
        return count

    def _find_singlet_pairs(self) -> List[Basepair]:
        """Find singlet pairs in the RNA structure.

        Returns:
            List of basepairs that are cWW but don't have cWW pairs above or below them
        """
        singlet_pairs = []
        count = 0
        for bp in self.cww_basepairs:
            count += 1
            if self._number_of_flanking_bps(bp) == 0:
                singlet_pairs.append(bp)

        return singlet_pairs

    def _do_residues_contain_singlet_pair(
        self,
        residues: List[Residue],
    ) -> bool:
        for res in residues:
            key = f"{res.get_str()}-{res.get_str()}"
            if key in self.singlet_pairs_lookup:
                return True
        return False

    def _assign_end_basepairs(
        self, strands: List[List[Residue]], basepairs: List[Basepair]
    ) -> List[Basepair]:
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_str())
            end_residue_ids.append(s[-1].get_str())

        # First collect all potential end basepairs
        end_basepairs = []
        for bp in basepairs:
            if (
                bp.res_1.get_str() in end_residue_ids
                and bp.res_2.get_str() in end_residue_ids
            ):
                end_basepairs.append(bp)

        # Track basepairs by residue
        residue_basepairs = {}
        for bp in end_basepairs:
            res1 = bp.res_1.get_str()
            res2 = bp.res_2.get_str()
            if res1 not in residue_basepairs:
                residue_basepairs[res1] = []
            if res2 not in residue_basepairs:
                residue_basepairs[res2] = []
            residue_basepairs[res1].append(bp)
            residue_basepairs[res2].append(bp)

        # Filter out weaker basepairs, keeping only the strongest one per residue pair
        filtered_basepairs = set()
        processed_residues = set()

        # Sort all basepairs by hbond score from strongest to weakest
        all_bps = []
        for bps in residue_basepairs.values():
            all_bps.extend(bps)
        all_bps.sort(key=lambda x: x.hbond_score, reverse=True)

        # Process basepairs in order of strength
        for bp in all_bps:
            res1 = bp.res_1.get_str()
            res2 = bp.res_2.get_str()

            # Only add if neither residue has been processed yet
            if res1 not in processed_residues and res2 not in processed_residues:
                filtered_basepairs.add(bp)
                processed_residues.add(res1)
                processed_residues.add(res2)

        return list(filtered_basepairs)

    def _get_basepairs_for_strands(
        self, strands: List[List[Residue]]
    ) -> List[Basepair]:
        basepairs = []
        residues = []
        for strand in strands:
            for res in strand:
                residues.append(res.get_str())
        for bp in self.basepairs:
            res1, res2 = self.chains.get_residues_in_basepair(bp)
            if res1 is None or res2 is None:
                continue
            if res1.get_str() in residues and res2.get_str() in residues:
                basepairs.append(bp)
        return basepairs

    def _get_basepair_ends_for_strands(
        self, strands: List[List[Residue]]
    ) -> List[Basepair]:
        basepairs = []
        end_ids = []
        for strand in strands:
            end_ids.append(strand[0].get_str())
            end_ids.append(strand[-1].get_str())
        for bp in self.cww_basepairs:
            if bp in basepairs:
                continue
            res1, res2 = self.chains.get_residues_in_basepair(bp)
            if res1.get_str() in end_ids and res2.get_str() in end_ids:
                basepairs.append(bp)
        return basepairs

    def _get_potential_basepair_ends_for_strands(
        self, strands: List[List[Residue]]
    ) -> List[Basepair]:
        basepairs = []
        for strand in strands:
            for res in strand:
                if res.get_str() in self.cww_residues_to_basepairs:
                    basepairs.append(self.cww_residues_to_basepairs[res.get_str()])
        return basepairs

    # hairpin functions ################################################################

    def get_looped_strands(self) -> List[List[Residue]]:
        """Find potential hairpin motifs in the RNA structure.

        Returns:
            List of Motif objects representing hairpins
        """
        distances = self._get_hairpin_distances()
        return self._get_shorest_hairpin_motifs(distances)

    def _get_hairpin_distances(self) -> Dict[str, int]:
        """Calculate distances between basepaired residues that could form hairpins."""
        distances = {}
        for bp in self.cww_basepairs_lookup.values():
            res1, res2 = self.chains.get_residues_in_basepair(bp)
            if not self.chains.are_residues_on_same_chain(res1, res2):
                continue
            if not self._is_next_residues_basepaired(res1, res2):
                continue
            chain = self.chains.get_chain_between_basepair(bp)
            if self._do_residues_contain_singlet_pair(chain):
                print("made it")
                continue
            distances[f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"] = len(chain)

        return distances

    def _is_next_residues_basepaired(
        self,
        res1: Residue,
        res2: Residue,
    ) -> bool:
        """Check if a basepair could form a valid hairpin."""
        chain_num_1, pos_1 = self.chains.get_residue_position(res1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res2)

        if pos_1 > pos_2:
            pos_1, pos_2 = pos_2, pos_1

        next_res_1 = self.chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
        next_res_2 = self.chains.get_residue_by_pos(chain_num_2, pos_2 - 1)

        # Check if adjacent residues are also basepaired
        key = f"{next_res_1.get_str()}-{next_res_2.get_str()}"
        return key not in self.cww_basepairs_lookup

    def _get_shorest_hairpin_motifs(self, distances: Dict[str, int]) -> List[Motif]:
        """Build Motif objects for valid hairpins."""
        sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
        hairpins = []
        used_residues = set()
        for pos, (pair_str, _) in enumerate(sorted_pairs):
            bp = self.cww_basepairs_lookup[pair_str]
            chain = self.chains.get_chain_between_basepair(bp)
            # Skip if overlaps with existing hairpin
            if any(res.get_str() in used_residues for res in chain):
                continue
            # Add residues to used set
            used_residues.update(res.get_str() for res in chain)
            basepairs = self._get_basepairs_for_strands([chain])

            hairpins.append(
                Motif(
                    f"HAIRPIN-{pos}",
                    "HAIRPIN",
                    "",
                    "",
                    "",
                    [chain],
                    basepairs,
                    [bp],
                    [],
                )
            )

        return hairpins

    # helix functions ##################################################################

    def _is_helix_start(
        self, bp: Basepair, cww_basepairs: Dict[str, Basepair], used_basepairs: Set[str]
    ) -> bool:
        """Check if basepair could start a new helix."""
        prev_pair = self._get_prev_pair_id(bp)
        if prev_pair is None:
            return True
        if prev_pair in cww_basepairs:
            return False
        else:
            return True

    def _get_prev_pair_id(self, bp: Basepair) -> Optional[str]:
        res_1, res_2 = self.chains.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.chains.get_residue_position(res_1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res_2)
        prev_res1 = self.chains.get_residue_by_pos(chain_num_1, pos_1 - 1)
        prev_res2 = self.chains.get_residue_by_pos(chain_num_2, pos_2 + 1)
        if prev_res1 is None or prev_res2 is None:
            return None
        return prev_res1.get_str() + "-" + prev_res2.get_str()

    def _get_next_pair_id(self, bp: Basepair) -> Optional[str]:
        res_1, res_2 = self.chains.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.chains.get_residue_position(res_1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res_2)
        next_res_1 = self.chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
        next_res_2 = self.chains.get_residue_by_pos(chain_num_2, pos_2 - 1)
        if next_res_1 is None or next_res_2 is None:
            return None
        return next_res_1.get_str() + "-" + next_res_2.get_str()

    def _build_helix(
        self,
        start_bp: Basepair,
        hairpin_x3nda_strs: List[str],
        used_basepairs: Set[str],
    ) -> Optional[Motif]:
        """Build a helix starting from the given basepair."""
        current_bp = start_bp

        # Follow chain of basepairs
        seen = []
        open_bp = [start_bp]
        helix = []

        while open_bp:
            current_bp = open_bp.pop(0)
            helix.append(current_bp)
            flanking_bps = self._get_possible_flanking_bps(current_bp)
            for res1, res2 in flanking_bps:
                if res1 is None or res2 is None:
                    continue
                key1 = f"{res1.get_str()}-{res2.get_str()}"
                key2 = f"{res2.get_str()}-{res1.get_str()}"
                if key1 not in self.cww_basepairs_lookup:
                    continue
                if key1 in used_basepairs or key1 in seen:
                    continue
                if key2 in used_basepairs or key2 in seen:
                    continue
                if (
                    res1.get_str() in hairpin_x3nda_strs
                    or res2.get_str() in hairpin_x3nda_strs
                ):
                    continue
                bp = self.cww_basepairs_lookup[key1]
                open_bp.append(bp)
                seen.append(key1)
                seen.append(key2)

        # Need at least 2 basepairs for a helix or is just a singlet
        if len(helix) < 2:
            return None

        residues = set()
        for bp in helix:
            key1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            key2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
            used_basepairs.add(key1)
            used_basepairs.add(key2)
            res1, res2 = self.chains.get_residues_in_basepair(bp)
            residues.add(res1)
            residues.add(res2)
        strands = get_rna_chains(list(residues))

        return Motif(
            "HELIX-{pos}",
            "HELIX",
            "",
            "",
            "",
            strands,
            helix,
            self._assign_end_basepairs(strands, helix),
            [],
        )

    def _do_all_residues_have_cww_basepairs(self, strands: List[List[Residue]]) -> bool:
        for strand in strands:
            for res in strand:
                if res.get_str() not in self.cww_residues_to_basepairs:
                    return False
        return True

    def _extend_strands_with_basepairs(
        self, strands: List[List[Residue]]
    ) -> List[List[Residue]]:
        extended_strands = []
        for strand in strands:
            first_res = strand[0]
            last_res = strand[-1]
            prev_res = self.chains.get_previous_residue_in_chain(first_res)
            if prev_res and prev_res.get_str() in self.cww_residues_to_basepairs:
                strand.insert(0, prev_res)
            next_res = self.chains.get_next_residue_in_chain(last_res)
            if next_res and next_res.get_str() in self.cww_residues_to_basepairs:
                strand.append(next_res)
            extended_strands.append(strand)
        return extended_strands

    # all non-helical motifs ##########################################################
    def get_non_helical_strands(self, helices: List[Motif]) -> List[List[Residue]]:
        """Find all non-helical strands in the RNA structure.

        Returns:
            List of lists of Residue objects representing non-helical strands
        """
        # Get all helical residues
        helical_residues = []
        for h in helices:
            helical_residues.extend(h.get_residues())

        strands = []
        for chain in self.chains.chains:
            current_strand = []
            for res in chain:
                if res not in helical_residues:
                    current_strand.append(res)
                else:
                    if len(current_strand) > 0:
                        strands.append(current_strand)
                        current_strand = []
            if len(current_strand) > 0:
                strands.append(current_strand)

        # Add flanking basepairs to strands this makes significantly easier to combine
        # strands together
        strands = self._extend_strands_with_basepairs(strands)

        return strands

    def _is_multi_strand_motif_valid(
        self, strands: List[List[Residue]], end_basepairs: List[Basepair]
    ) -> bool:
        end_bp_res = {}
        for i, bp in enumerate(end_basepairs):
            end_bp_res[bp.res_1.get_str()] = 0
            end_bp_res[bp.res_2.get_str()] = 0

        chain_ends_in_bp = {}
        for strand in strands:
            for res in [strand[0], strand[-1]]:
                chain_ends_in_bp[res.get_str()] = 0

        for strand in strands:
            for res in [strand[0], strand[-1]]:
                if res.get_str() in end_bp_res:
                    end_bp_res[res.get_str()] = 1
                    chain_ends_in_bp[res.get_str()] = 1

        if len(end_bp_res) != sum(end_bp_res.values()):
            return False
        if len(chain_ends_in_bp) != sum(chain_ends_in_bp.values()):
            return False

        return True

    def _find_connected_strands(
        self,
        unprocessed_strands: List[List[Residue]],
        current_motif_strands: List[List[Residue]],
        current_basepair_ends: List[Basepair],
    ) -> Tuple[List[List[Residue]], List[Basepair]]:
        """Find all strands that share basepairs with the current motif strands.

        Args:
            unprocessed_strands: List of strands not yet assigned to a motif
            current_motif_strands: List of strands in current motif
            current_basepair_ends: List of basepairs at ends of current motif strands
            bp_dict: Dictionary mapping residue strings to basepair objects

        Returns:
            Tuple containing:
                - Updated list of strands in current motif
                - Updated list of basepairs at strand ends
        """
        changed = True
        while changed:
            changed = False
            # Look through remaining strands
            for strand in unprocessed_strands:
                if strand in current_motif_strands:
                    continue
                shares_bp = False
                # Check first and last residue in strand for basepairs with motif
                for res in [strand[0], strand[-1]]:
                    for end in current_basepair_ends:
                        if (
                            res.get_str() == end.res_1.get_str()
                            or res.get_str() == end.res_2.get_str()
                        ):
                            shares_bp = True
                            break

                # If strand shares bp, add it to current motif
                if shares_bp:
                    current_motif_strands.append(strand)
                    for res in [strand[0], strand[-1]]:
                        if res.get_str() in self.cww_residues_to_basepairs:
                            bp = self.cww_residues_to_basepairs[res.get_str()]
                            if bp not in current_basepair_ends:
                                current_basepair_ends.append(bp)
                    changed = True

        return current_motif_strands, current_basepair_ends

    def _get_non_canonical_motif_from_strands(
        self,
        initial_strand: List[Residue],
        count: int,
        unprocessed_strands: List[List[Residue]],
    ) -> List[Motif]:
        current_motif_strands = [initial_strand]
        current_basepair_ends = []
        for res in [initial_strand[0], initial_strand[-1]]:
            if res.get_str() in self.cww_residues_to_basepairs:
                bp = self.cww_residues_to_basepairs[res.get_str()]
                if bp not in current_basepair_ends:
                    current_basepair_ends.append(bp)

        current_motif_strands, current_basepair_ends = self._find_connected_strands(
            unprocessed_strands,
            current_motif_strands,
            current_basepair_ends,
        )

        for strand in current_motif_strands:
            if strand in unprocessed_strands:
                unprocessed_strands.remove(strand)

        if len(current_motif_strands) > 1:
            if not self._is_multi_strand_motif_valid(
                current_motif_strands, current_basepair_ends
            ):
                pos = 0
                # break strands back up
                motifs = []
                for strand in current_motif_strands:
                    motifs.append(
                        Motif(
                            f"UNKNOWN-{count}-{pos}",
                            "UNKNOWN",
                            "",
                            "",
                            "",
                            [strand],
                            self._get_basepairs_for_strands([strand]),
                            self._get_basepair_ends_for_strands([strand]),
                            [],
                        )
                    )
                    pos += 1
                return motifs

        return [
            Motif(
                f"UNKNOWN-{count}",
                "UNKNOWN",
                "",
                "",
                "",
                current_motif_strands,
                self._get_basepairs_for_strands(current_motif_strands),
                self._get_basepair_ends_for_strands(current_motif_strands),
                [],
            )
        ]

    # finalizing motifs ################################################################
    def _name_motif(self, motif: Motif) -> str:
        name = f"{motif.mtype}-{motif.size}-{motif.sequence}-{self.pdb_id}"
        if name not in self.motif_name_count:
            self.motif_name_count[name] = 1
        else:
            self.motif_name_count[name] += 1
        name += f"-{self.motif_name_count[name]}"
        return name

    def _get_motif_sequence(self, motif: Motif) -> str:
        seqs = []
        for strand in motif.strands:
            s = self._get_strand_sequence(strand)
            seqs.append(s)
        return "-".join(seqs)

    def _get_motif_topology(self, motif: Motif) -> str:
        if motif.mtype == "HELIX":
            return len(motif.strands[0])
        elif motif.mtype == "SSTRAND":
            return len(motif.strands[0])
        else:
            return "-".join(str(len(strand) - 2) for strand in motif.strands)

    def _get_strand_sequence(self, strand: List[Residue]) -> str:
        seq = ""
        for res in strand:
            if res.res_id in ["A", "G", "C", "U"]:
                seq += res.res_id
            else:
                seq += "X"
        return seq

    def _standardize_strands(self, strands: List[List[Residue]]) -> List[List[Residue]]:
        """Order strands by length (longest first) and then by string comparison.

        Args:
            strands: List of strands (lists of Residue objects)

        Returns:
            Ordered list of strands
        """
        # Sort strands by length (descending) and then by string comparison of first residue
        return sorted(strands, key=lambda x: (-len(x), self._get_strand_sequence(x)))

    def _assign_motif_type(self, motif: Motif) -> str:
        num_loop_strands = 0
        for strand in motif.strands:
            if self._is_strand_a_loop(strand, motif.basepair_ends):
                num_loop_strands += 1
        if len(motif.basepair_ends) == 0 and num_loop_strands == 0:
            return "SSTRAND"
        elif len(motif.basepair_ends) == 1 and num_loop_strands == 1:
            return "HAIRPIN"
        elif (
            len(motif.basepair_ends) == 2
            and len(motif.strands) == 2
            and self._do_all_residues_have_cww_basepairs(motif.strands)
        ):
            return "HELIX"
        elif len(motif.basepair_ends) == 2 and len(motif.strands) == 2:
            return "TWOWAY"
        elif len(motif.basepair_ends) > 2 and num_loop_strands == 0:
            return "NWAY"
        else:
            return "UNKNOWN"

    def _finalize_motif(self, motif: Motif) -> Motif:
        if motif.mtype == "UNKNOWN":
            motif.mtype = self._assign_motif_type(motif)
        # remove strand ends that are basepairs for single stranded motifs
        if motif.mtype == "SSTRAND":
            strand = motif.strands[0]
            if strand[0].get_str() in self.helical_residues:
                if len(strand) > 1:
                    strand = strand[1:]
                else:
                    strand = []
            if len(strand) == 0:
                return None
            if strand[-1].get_str() in self.helical_residues:
                if len(strand) > 1:
                    strand = strand[:-1]
                else:
                    strand = []
            if len(strand) == 0:
                return None
            motif.strands = [strand]
        motif.strands = self._standardize_strands(motif.strands)
        motif.sequence = self._get_motif_sequence(motif)
        # just to make sure all basepairs are in the motif
        motif.basepairs = self._get_basepairs_for_strands(motif.strands)
        motif.end_basepairs = self._get_basepair_ends_for_strands(motif.strands)
        motif.size = self._get_motif_topology(motif)
        motif.name = self._name_motif(motif)
        res_strs = [res.get_str() for res in motif.get_residues()]
        for hbond in self.hbonds:
            if hbond.res_1.get_str() in res_strs or hbond.res_2.get_str() in res_strs:
                motif.hbonds.append(hbond)
        return motif

    def _does_motif_have_chain_ends(self, motif: Motif) -> bool:
        motif_chain_ends = []
        for strand in motif.strands:
            motif_chain_ends.extend([strand[0], strand[-1]])
        for res in motif_chain_ends:
            if self.chains.is_chain_end(res):
                return True
        return False

    def _is_strand_a_loop(
        self, strand: List[Residue], end_basepairs: List[Basepair]
    ) -> bool:
        end_residue_ids = [strand[0].get_str(), strand[-1].get_str()]
        for bp in end_basepairs:
            if (
                bp.res_1.get_str() in end_residue_ids
                and bp.res_2.get_str() in end_residue_ids
            ):
                return True
        return False

    def _find_and_assign_finished_motifs(
        self, motifs: List[Motif]
    ) -> Tuple[List[Motif], List[Motif]]:
        finished_motifs = []
        unfinished_motifs = []
        for motif in motifs:
            # if it has multiple strands then its finished
            if len(motif.strands) > 1:
                finished_motifs.append(motif)
                continue
            # check single strand which could be a chain end
            if self._does_motif_have_chain_ends(motif):
                motif.mtype = "SSTRAND"
                finished_motifs.append(motif)
                continue
            elif self._is_strand_a_loop(motif.strands[0], motif.basepair_ends):
                motif.mtype = "HAIRPIN"
                finished_motifs.append(motif)
                continue
            unfinished_motifs.append(motif)
        return finished_motifs, unfinished_motifs

    def get_bulge_or_multiway_junction(
        self,
        current_strand: List[Residue],
        strands_between_helices: List[List[Residue]],
    ) -> Motif:
        current_motif_strands = [current_strand]
        current_basepair_ends = []
        for strand in current_motif_strands:
            for res in [strand[0], strand[-1]]:
                if res.get_str() in self.cww_residues_to_basepairs:
                    bp = self.cww_residues_to_basepairs[res.get_str()]
                    if bp not in current_basepair_ends:
                        current_basepair_ends.append(bp)

        current_motif_strands, current_basepair_ends = self._find_connected_strands(
            strands_between_helices,
            current_motif_strands,
            current_basepair_ends,
        )
        if len(current_motif_strands) == 1:
            return None

        if not self._is_multi_strand_motif_valid(
            current_motif_strands, current_basepair_ends
        ):
            return None

        return Motif(
            "UNKNOWN-{pos}",
            "UNKNOWN",
            "",
            "",
            "",
            current_motif_strands,
            self._get_basepairs_for_strands(current_motif_strands),
            current_basepair_ends,
            [],
        )

    # debug functions #################################################################

    def _get_end_basepair_overlaps(
        self, motif: Motif, motifs: List[Motif]
    ) -> List[Motif]:
        motif_ends = []
        for end in motif.basepair_ends:
            motif_ends.append(end.res_1.get_str())
            motif_ends.append(end.res_2.get_str())
        for m in motifs:
            if m.name == motif.name:
                continue
            for strand in m.strands:
                for res in [strand[0], strand[-1]]:
                    if res.get_str() in motif_ends:
                        m.to_cif()
                        print(f"Residue: {res.get_str()} in motif: {m.name}")

    def inspect_motif(self, motif: Motif, motifs: List[Motif] = None):
        print(f"Motif: {motif.name}")
        print(f"Type: {motif.mtype}")
        print(f"Size: {motif.size}")
        print(f"Sequence: {motif.sequence}")
        print(f"Num Strands: {motif.num_strands()}")
        print(f"Num Basepairs: {motif.num_basepairs()}")
        print(f"Num Basepair Ends: {motif.num_basepair_ends()}")
        for strand in motif.strands:
            for res in [strand[0], strand[-1]]:
                if res.get_str() in self.cww_residues_to_basepairs:
                    bp = self.cww_residues_to_basepairs[res.get_str()]
                    print(
                        f"Residue: {res.get_str()} in End Basepair: {bp.res_1.get_str()} {bp.res_2.get_str()}"
                    )
        print(f"Num Hbonds: {len(motif.hbonds)}")
        for hbond in motif.hbonds:
            print(
                f"Hbond: {hbond.res_1.get_str()} {hbond.res_2.get_str()} {hbond.atom_1} {hbond.atom_2} {hbond.res_type_1} {hbond.res_type_2} {hbond.distance} {hbond.angle_1} {hbond.angle_2} {hbond.dihedral_angle} {hbond.score}"
            )
        if motifs is None:
            return
        print("overlaps with other motifs:")
        self._get_end_basepair_overlaps(motif, motifs)
        for strand in motif.strands:
            for res in strand:
                print(res.get_str(), end=" ")
            print()

        if motif.mtype != "SSTRAND":
            return
        print("extending strands with basepairs")
        motif.strands = self._extend_strands_with_basepairs(motif.strands)
        motif.basepair_ends = self._get_potential_basepair_ends_for_strands(
            motif.strands
        )
        for strand in motif.strands:
            for res in strand:
                print(res.get_str(), end=" ")
            print()
        self._get_end_basepair_overlaps(motif, motifs)

    def inspect_motif_neighbors(self, motif: Motif, motifs: List[Motif]):
        pass

    def find_motif_interactions(
        self, motifs: List[Motif]
    ) -> List[Tuple[Motif, Motif, int]]:
        """Find pairs of motifs that share basepairs between their residues.

        Args:
            motifs: List of motifs to check for interactions

        Returns:
            List of tuples containing pairs of interacting motifs and number of shared basepairs
        """
        interactions = []
        # Check each pair of motifs
        for i, motif1 in enumerate(motifs):
            for motif2 in motifs[i + 1 :]:
                # Get all residues in each motif
                residues1 = set(motif1.get_residues())
                residues2 = set(motif2.get_residues())

                # Count basepairs between the motifs
                num_shared_bps = 0
                for bp in self.basepairs:
                    res1 = self.chains.get_residue_by_str(bp.res_1.get_str())
                    res2 = self.chains.get_residue_by_str(bp.res_2.get_str())
                    if (res1 in residues1 and res2 in residues2) or (
                        res1 in residues2 and res2 in residues1
                    ):
                        num_shared_bps += 1

                if num_shared_bps > 0:
                    interactions.append((motif1, motif2, num_shared_bps))
        return interactions


# Motifs from DSSR ###################################################################


class MotifFactoryFromOther:
    """Class for processing motifs and interactions from PDB files"""

    def __init__(
        self,
        pdb_name: str,
        chains: Chains,
        residues: List[Residue],
        basepairs: List[Basepair],
    ):
        """
        Initialize the MotifProcessor

        Args:
            count (int): # of PDBs processed (loaded from outside)
            pdb_path (str): path to the source PDB
        """
        self.pdb_name = pdb_name
        self.residues = residues
        self.basepairs = basepairs
        self.chains = chains
        self.cww_basepairs = self._get_cww_basepairs(basepairs)
        self.used_names = {}

    def get_motifs_from_dssr(self) -> List[Motif]:
        """
        Process the PDB file and extract motifs and interactions

        Returns:
            motif_list (list): list of motifs
        """
        log.debug(f"{self.pdb_name}")
        new_residues = {k: Residue.from_dict(v) for k, v in self.residues.items()}
        all_residues = {}
        for res in new_residues.values():
            all_residues[res.get_x3dna_str()] = res

        json_path = os.path.join(DATA_PATH, "dssr_output", f"{self.pdb_name}.json")
        dssr_output = DSSROutput(json_path=json_path)
        dssr_motifs = dssr_output.get_motifs()
        motifs = []
        log.info(f"Processing {len(dssr_motifs)} motifs")
        for m in dssr_motifs:
            residues = self._get_residues_for_motif(m, all_residues)
            m = self._generate_motif(m.mtype, residues)
            motifs.append(m)
        log.info(f"Final number of motifs: {len(motifs)}")
        return motifs

    def get_motif_from_atlas(self, atlas_mtype, res_strs):
        residues = []
        for res_str in res_strs:
            if res_str in self.residues:
                residues.append(self.residues[res_str])
            else:
                log.warning(f"Residue {res_str} not found in {self.pdb_name}")
        return self._generate_motif(atlas_mtype, residues)

    def from_residues(self, residues: List[Residue]) -> Motif:
        return self._generate_motif(residues)

    def _get_residues_for_motif(
        self, m, all_residues: Dict[str, Residue]
    ) -> List[Residue]:
        residues = []
        for nt in m.nts_long:
            if nt in all_residues:
                residues.append(all_residues[nt])
            else:
                log.warning(f"Residue {nt} not found in {self.pdb_name}")
                continue
        return residues

    def _get_cww_basepairs(self, basepairs: List[Basepair]) -> Dict[str, Basepair]:
        """Get dictionary of cWW basepairs keyed by residue pair strings."""
        allowed_pairs = []
        f = open(os.path.join(RESOURCES_PATH, "valid_cww_pairs.txt"))
        lines = f.readlines()
        for line in lines:
            allowed_pairs.append(line.strip())
        f.close()
        cww_basepairs = {}
        two_hbond_pairs = ["A-U", "U-A", "G-U", "U-G"]
        three_hbond_pairs = ["G-C", "C-G"]
        for bp in basepairs:
            if bp.lw != "cWW" or bp.bp_type not in allowed_pairs:
                continue
            if (
                self.chains.get_residue_by_str(bp.res_1.get_str()) is None
                or self.chains.get_residue_by_str(bp.res_2.get_str()) is None
            ):
                continue
            # stops a lot of bad basepairs from being included
            if bp.bp_type in two_hbond_pairs and bp.hbond_score < 1.3:
                continue
            if bp.bp_type in three_hbond_pairs and bp.hbond_score < 2.0:
                continue
            key1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            key2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
            cww_basepairs[key1] = bp
            cww_basepairs[key2] = bp
        return cww_basepairs

    def _assign_end_basepairs(self, strands: List[List[Residue]]) -> List[Basepair]:
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_str())
            end_residue_ids.append(s[-1].get_str())

        # First collect all potential end basepairs
        end_basepairs = []
        for bp in self.basepairs:
            if (
                not bp.res_1.get_str() in end_residue_ids
                or not bp.res_2.get_str() in end_residue_ids
            ):
                continue
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in self.cww_basepairs:
                end_basepairs.append(self.cww_basepairs[key])
        # Track basepairs by residue
        residue_basepairs = {}
        for bp in end_basepairs:
            res1 = bp.res_1.get_str()
            res2 = bp.res_2.get_str()
            if res1 not in residue_basepairs:
                residue_basepairs[res1] = []
            if res2 not in residue_basepairs:
                residue_basepairs[res2] = []
            residue_basepairs[res1].append(bp)
            residue_basepairs[res2].append(bp)

        # Filter out weaker basepairs, keeping only the strongest one per residue pair
        filtered_basepairs = set()
        processed_residues = set()

        # Sort all basepairs by hbond score from strongest to weakest
        all_bps = []
        for bps in residue_basepairs.values():
            all_bps.extend(bps)
        all_bps.sort(key=lambda x: x.hbond_score, reverse=True)

        # Process basepairs in order of strength
        for bp in all_bps:
            res1 = bp.res_1.get_str()
            res2 = bp.res_2.get_str()

            # Only add if neither residue has been processed yet
            if res1 not in processed_residues and res2 not in processed_residues:
                filtered_basepairs.add(bp)
                processed_residues.add(res1)
                processed_residues.add(res2)

        return list(filtered_basepairs)

    def _generate_motif(self, x3dna_mtype: str, residues: List[Residue]) -> Motif:
        # We need to determine the data for the motif and build a class
        # First get the type
        mtype = "UNKNOWN"
        if x3dna_mtype == "SINGLE_STRAND":
            mtype = "SSTRAND"
        elif x3dna_mtype == "HAIRPIN":
            mtype = "HAIRPIN"
        elif x3dna_mtype == "STEM":
            mtype = "HELIX"
        elif x3dna_mtype == "ILOOP":
            mtype = "TWOWAY"
        elif x3dna_mtype == "BULGE":
            mtype = "TWOWAY"
        elif x3dna_mtype == "JUNCTION":
            mtype = "NWAY"
        residue_ids = [res.get_str() for res in residues]
        strands = get_rna_chains(residues)
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_str())
            end_residue_ids.append(s[-1].get_str())
        basepairs = []
        for bp in self.basepairs:
            if bp.res_1.get_str() in residue_ids and bp.res_2.get_str() in residue_ids:
                basepairs.append(bp)
        end_basepairs = self._assign_end_basepairs(strands)
        sequence = self._find_sequence(strands).replace("&", "-")
        mname = f"{mtype}-{sequence}-{self.pdb_name}"
        if mname in self.used_names:
            self.used_names[mname] += 1
        else:
            self.used_names[mname] = 1
        mname += f"-{self.used_names[mname]}"
        return Motif(
            mname,
            mtype,
            self.pdb_name,
            "",
            sequence,
            strands,
            basepairs,
            end_basepairs,
            [],
        )

    def _find_sequence(self, strands_of_rna: List[List[Residue]]) -> str:
        """Find sequences from found strands of RNA"""
        res_strands = []
        for strand in strands_of_rna:
            res_strand = []
            for residue in strand:
                mol_name = residue.res_id
                if mol_name in ["A", "G", "C", "U"]:
                    res_strand.append(mol_name)
                else:
                    res_strand.append("X")
            strand_sequence = "".join(res_strand)
            res_strands.append(strand_sequence)
        sequence = "&".join(res_strands)
        return sequence


def do_number_of_strands_match_mtype(mtype: str, strands: List[List[Residue]]):
    if mtype == "HAIRPIN":
        return len(strands) == 1
    elif mtype == "HELIX":
        return len(strands) == 2
    elif mtype == "TWOWAY":
        return len(strands) == 2
    elif mtype == "NWAY":
        return len(strands) > 2
    elif mtype == "SSTRAND":
        return len(strands) == 1
    else:
        return True


def do_number_of_basepairs_match_mtype(mtype: str, basepairs: List[Basepair]):
    if mtype == "HAIRPIN":
        return len(basepairs) == 1
    elif mtype == "HELIX":
        return len(basepairs) == 2
    elif mtype == "TWOWAY":
        return len(basepairs) == 2
    elif mtype == "NWAY":
        return len(basepairs) > 2
    elif mtype == "SSTRAND":
        return len(basepairs) == 0
    else:
        return True


# Motifs from Atlas #################################################################


@dataclass
class ResidueId:
    pdb_id: str
    model: str
    chain: str
    residue_type: str
    residue_num: int
    insert_code: str

    @staticmethod
    def _parse_triple_pipe_string(residue_str: str):
        spl_1 = residue_str.split("|||")
        if len(spl_1) != 1:
            insert_code = spl_1[1]
            # dont understand why this is sometimes not an insertion code
            if len(insert_code) > 1:
                insert_code = ""
        else:
            insert_code = ""

    @classmethod
    def from_string(cls, residue_str: str):
        # Split on | delimiter
        pdb_id, model, chain, residue_type, residue_num = residue_str.split("|")[0:5]
        insert_code = cls._parse_triple_pipe_string(residue_str)
        spl_2 = residue_str.split("||")
        if insert_code != "" and len(spl_2) != 1:
            if len(spl_2[1]) > 1:
                insert_code = spl_2[1][-1]
            else:
                insert_code = ""
        else:
            insert_code = ""
        return cls(
            pdb_id=pdb_id,
            model=model,
            chain=chain,
            residue_type=residue_type,
            residue_num=int(residue_num),
            insert_code=insert_code,
        )

    def to_x3dna_residue(self) -> X3DNAResidue:
        return X3DNAResidue(
            chain_id=self.chain,
            res_id=self.residue_type,
            num=self.residue_num,
            ins_code=self.insert_code,
            rtype=self.residue_type,
        )

    def get_str(self):
        return f"{self.chain}-{self.residue_type}-{self.residue_num}-{self.insert_code}"


def parse_atlas_csv(csv_path):
    f = open(csv_path, "r")
    group = None
    mtype = None
    data = []
    if os.path.basename(csv_path).startswith("hl"):
        mtype = "HAIRPIN"
    elif os.path.basename(csv_path).startswith("il"):
        mtype = "TWOWAY"
    elif os.path.basename(csv_path).startswith("j3"):
        mtype = "NWAY"
    else:
        raise ValueError(f"Unknown motif type: {os.path.basename(csv_path)}")
    for line in f:
        if line.startswith(">"):
            group = line.strip()[1:]
            continue
        residue_infos = line.strip().split(",")
        x3dna_res = []
        pdb_id = None
        for residue_info in residue_infos:
            # remove quotes
            residue_info = residue_info[1:-1]
            residue_id = ResidueId.from_string(residue_info)
            x3dna_res.append(residue_id.to_x3dna_residue().get_str())
            pdb_id = residue_id.pdb_id
        data.append(
            {
                "pdb_id": pdb_id,
                "group": group,
                "mtype": mtype,
                "residues": x3dna_res,
            }
        )
    return pd.DataFrame(data)


def get_data_from_motif(m: Motif, pdb_id: str, has_singlet_flank: bool):
    return {
        "pdb_id": pdb_id,
        "motif": m.name,
        "mtype": m.mtype,
        "n_strands": len(m.strands),
        "n_basepairs": len(m.basepairs),
        "n_basepair_ends": len(m.basepair_ends),
        "n_residues": len(m.get_residues()),
        "residues": [r.get_str() for r in m.get_residues()],
        "correct_n_strands": do_number_of_strands_match_mtype(m.mtype, m.strands),
        "correct_n_basepairs": do_number_of_basepairs_match_mtype(
            m.mtype, m.basepair_ends
        ),
        "has_singlet_flank": has_singlet_flank,
    }


class MotifSetComparerer:
    def compare_motifs(self, pdb_id, df):
        try:
            motifs = get_cached_motifs(pdb_id)
        except Exception as e:
            print(f"Error getting motifs for {pdb_id}: {e}")
            return None
        print(f"Processing {pdb_id}")
        # Get tertiary contacts info
        in_tc = self._get_tertiary_contacts(pdb_id)
        # Get motif mappings
        res_to_motif_id = get_res_to_motif_mapping(motifs)
        motifs_by_name = {m.name: m for m in motifs}
        seen = []
        # Initialize columns
        dssr_motifs = self._initialize_dssr_motifs_df(df)
        # Compare motifs
        dssr_motifs, seen = self._compare_motifs(dssr_motifs, motifs, seen)
        # Find overlapping motifs
        dssr_motifs = self._find_overlapping_motifs(
            dssr_motifs, res_to_motif_id, motifs_by_name, in_tc
        )
        # Add missing motifs
        dssr_motifs = self._add_missing_motifs(dssr_motifs, motifs, seen, pdb_id)
        return dssr_motifs

    def _get_tertiary_contacts(self, pdb_id):
        """
        Maps motifs in tertiary contacts to their residues
        """
        in_tc = {}
        tert_path = os.path.join(
            DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json"
        )
        if file_exists_and_has_content(tert_path):
            df_tert = pd.read_json(tert_path)
        else:
            df_tert = pd.DataFrame()
        for i, row in df_tert.iterrows():
            if row["motif_1"] not in in_tc:
                in_tc[row["motif_1"]] = []
            in_tc[row["motif_1"]].extend(row["motif_1_res"])
            if row["motif_2"] not in in_tc:
                in_tc[row["motif_2"]] = []
            in_tc[row["motif_2"]].extend(row["motif_2_res"])
        return in_tc

    def _initialize_dssr_motifs_df(self, dssr_motifs):
        dssr_motifs = dssr_motifs.copy()
        dssr_motifs["found"] = False  # Whether the motif was found in our database
        dssr_motifs["misclassified"] = False  # Whether the motif was misclassified
        dssr_motifs["missing"] = False  # Whether the motif was missing
        dssr_motifs["overlapping_motifs"] = [
            [] for _ in range(len(dssr_motifs))
        ]  # Motifs that overlap with the motif
        dssr_motifs["contained_in_motifs"] = [
            [] for _ in range(len(dssr_motifs))
        ]  # Motifs that are contained in the motif
        dssr_motifs["in_tc"] = False  # Whether the motif is in tertiary contacts
        return dssr_motifs

    def _compare_motifs(self, dssr_motifs, motifs, seen):
        for i, row in dssr_motifs.iterrows():
            for m in motifs:
                if len(m.get_residues()) != len(row["residues"]):
                    continue
                res = sorted([r.get_str() for r in m.get_residues()])
                res_dssr = sorted(row["residues"])
                if res != res_dssr:
                    continue
                dssr_motifs.at[i, "found"] = True
                seen.append(m.name)
                if m.mtype != row["mtype"]:
                    dssr_motifs.at[i, "misclassified"] = True
                dssr_motifs.at[i, "overlapping_motifs"].append(m.name)
        return dssr_motifs, seen

    def _find_overlapping_motifs(
        self, dssr_motifs, res_to_motif_id, motifs_by_name, in_tc
    ):
        for i, row in dssr_motifs.iterrows():
            if row["found"]:
                continue
            overlapping_motifs = []
            potential_tc_res = []
            for r in row["residues"]:
                if r in res_to_motif_id:
                    m_name = res_to_motif_id[r]
                    if m_name not in overlapping_motifs:
                        overlapping_motifs.append(m_name)
                    if m_name in in_tc:
                        potential_tc_res.extend(in_tc[m_name])
            for r in row["residues"]:
                if r in potential_tc_res:
                    dssr_motifs.at[i, "in_tc"] = True
            dssr_motifs.at[i, "overlapping_motifs"] = overlapping_motifs
            for m in overlapping_motifs:
                overlap_m = motifs_by_name[m]
                overlap_m_res = [r.get_str() for r in overlap_m.get_residues()]
                if all(r in overlap_m_res for r in row["residues"]):
                    dssr_motifs.at[i, "contained_in_motifs"].append(m)
        return dssr_motifs

    def _add_missing_motifs(self, dssr_motifs, motifs, seen, pdb_id):
        data = []
        for m in motifs:
            if m.name in seen:
                continue
            data.append(
                {
                    "pdb_id": pdb_id,
                    "motif": m.name,
                    "mtype": m.mtype,
                    "n_strands": len(m.strands),
                    "n_basepairs": len(m.basepairs),
                    "n_basepair_ends": len(m.basepair_ends),
                    "n_residues": len(m.get_residues()),
                    "residues": [r.get_str() for r in m.get_residues()],
                    "correct_n_strands": True,
                    "correct_n_basepairs": True,
                    "missing": True,
                    "misclassified": False,
                    "found": True,
                    "overlapping_motifs": [m.name],
                    "has_singlet_flank": False,
                    "contained_in_motifs": [],
                    "in_tc": False,
                }
            )
        df_missing = pd.DataFrame(data)
        return pd.concat([dssr_motifs, df_missing])


# comparing motifs ################################################################


def get_motifs_included_in_chains(motifs: List[Motif], chain_ids: List[str]):
    keep_motifs = []
    for m in motifs:
        keep = True
        for r in m.get_residues():
            if r.chain_id not in chain_ids:
                keep = False
                break
        if keep:
            keep_motifs.append(m)
    return keep_motifs


def find_duplicate_motifs(motifs, other_motifs):
    """Check for duplicate motifs between two sets of motifs.

    Args:
        motifs: List of reference motifs to compare against
        other_motifs: List of motifs to check for duplicates
    Returns:
        DataFrame containing duplicate information for each motif in other_motifs
    """
    data = []
    used_motifs = []
    for other_motif in other_motifs:
        result = find_best_matching_motif(other_motif, motifs, used_motifs)
        data.append(
            {
                "motif": other_motif.name,
                "repr_motif": result["best_motif"],
                "rmsd": result["best_rmsd"],
                "is_duplicate": result["is_duplicate"],
                "from_repr": True,
            }
        )
        if result["is_duplicate"]:
            used_motifs.append(result["best_motif"])
    return pd.DataFrame(data)


def find_best_matching_motif(query_motif, ref_motifs, used_motifs):
    """Find the best matching reference motif for a query motif.

    Args:
        query_motif: Motif to find match for
        ref_motifs: List of reference motifs to compare against
        used_motifs: List of already matched reference motifs

    Returns:
        Dict containing best matching motif and match statistics
    """
    best_motif = None
    best_rmsd = 1000
    query_coords = query_motif.get_c1prime_coords()

    if len(query_coords) < 2:
        return {"best_motif": None, "best_rmsd": best_rmsd, "is_duplicate": False}

    for ref_motif in ref_motifs:
        # Skip if motif already used or sequences don't match
        if ref_motif.name in used_motifs:
            continue
        if ref_motif.sequence != query_motif.sequence:
            continue
        try:
            ref_coords = ref_motif.get_c1prime_coords()
            if len(ref_coords) != len(query_coords):
                continue
            rotated_coords = superimpose_structures(ref_coords, query_coords)
            rmsd_val = rmsd(query_coords, rotated_coords)
            if rmsd_val < best_rmsd:
                best_rmsd = rmsd_val
                best_motif = ref_motif.name
        except:
            print("issues", ref_motif.name, query_motif.name)
            continue
    # Determine if match is close enough to be a duplicate
    is_duplicate = best_rmsd < 0.20 * len(query_coords)
    return {
        "best_motif": best_motif,
        "best_rmsd": best_rmsd,
        "is_duplicate": is_duplicate,
    }


def create_representative_motif_dataframe(repr_motifs, pdb_id, from_repr=True):
    repr_data = []
    for m in repr_motifs:
        repr_data.append(
            {
                "motif": m.name,
                "repr_motif": None,
                "rmsd": None,
                "is_duplicate": False,
                "repr_pdb": pdb_id,
                "child_pdb": None,
                "from_repr": from_repr,
            }
        )
    return pd.DataFrame(repr_data)


def _process_child_entry(child_entry, repr_motifs):
    """Process a single child entry to find duplicates with representative motifs.

    Args:
        child_entry: Child structure entry to process
        repr_motifs: List of motifs from representative structure

    Returns:
        tuple: (matches_df, unmatched_motifs) where matches_df contains duplicate matches
        and unmatched_motifs contains motifs that didn't match the representative
    """
    # Get motifs from child structure
    try:
        motifs = get_cached_motifs(child_entry.pdb_id)
    except:
        return None, []
    child_motifs = get_motifs_included_in_chains(motifs, child_entry.chain_ids)
    # Find duplicates between representative and child motifs
    duplicates_df = find_duplicate_motifs(repr_motifs, child_motifs)
    if len(duplicates_df) == 0:
        return None, []
    # Track motifs that didn't match representative
    unmatched = duplicates_df.query("rmsd == 1000.0")["motif"].values
    unmatched_set = [m for m in child_motifs if m.name in unmatched]
    # Keep matches with valid RMSD
    matches = duplicates_df.query("rmsd < 1000").copy()
    return matches, unmatched_set


def _align_to_other_entry_members(unmatched_motifs):
    """Compare unmatched motifs against each other to find additional duplicates.

    Args:
        unmatched_motifs: List of lists of unmatched motifs

    Returns:
        list: DataFrames containing duplicate information
    """
    results = []
    for i, motif_set_1 in enumerate(unmatched_motifs):
        if not motif_set_1:
            continue
        pdb_id = parse_motif_name(motif_set_1[0].name)[-1]
        # Compare against all subsequent unmatched sets
        for j, motif_set_2 in enumerate(unmatched_motifs[i + 1 :], i + 1):
            if not motif_set_2:
                continue
            child_pdb_id = parse_motif_name(motif_set_2[0].name)[-1]
            # Find duplicates between unmatched sets
            duplicates = find_duplicate_motifs(motif_set_1, motif_set_2)
            duplicates = duplicates.query("is_duplicate == True")
            duplicates["child_pdb"] = child_pdb_id
            duplicates["repr_pdb"] = pdb_id
            results.append(duplicates)
            # Remove found duplicates from second set
            duplicate_names = duplicates["motif"].values
            unmatched_motifs[j] = [
                m for m in motif_set_2 if m.name not in duplicate_names
            ]
        # Add remaining unmatched motifs from first set
        results.append(
            create_representative_motif_dataframe(motif_set_1, pdb_id, from_repr=False)
        )
    return results


def find_unique_motifs_in_non_redundant_set_entry(args):
    """Find unique motifs in a non-redundant set entry.

    Args:
        args: Tuple containing (set_id, repr_entry, child_entries)
            set_id: ID of the non-redundant set
            repr_entry: Representative structure entry
            child_entries: List of child structure entries

    Returns:
        Path to saved CSV file containing duplicate information, or None if file exists
    """
    set_id, repr_entry, child_entries = args
    # Check if output file already exists
    output_path = os.path.join(
        DATA_PATH, "dataframes", "non_redundant_sets", f"{set_id}.csv"
    )
    if os.path.exists(output_path):
        return None
    print(f"Processing set {set_id}")
    # Get motifs from representative structure
    motifs = get_cached_motifs(repr_entry.pdb_id)
    repr_motifs = get_motifs_included_in_chains(motifs, repr_entry.chain_ids)
    # Initialize results with representative motifs
    results = [
        create_representative_motif_dataframe(
            repr_motifs, repr_entry.pdb_id, from_repr=True
        )
    ]
    unmatched_motifs = []  # Motifs that don't match representative structure
    # Process each child structure
    for child_entry in child_entries:
        matches, unmatched = _process_child_entry(child_entry, repr_motifs)
        if matches is not None:
            matches["child_pdb"] = child_entry.pdb_id
            matches["repr_pdb"] = repr_entry.pdb_id
            results.append(matches)
        if unmatched:
            unmatched_motifs.append(unmatched)
    # Compare unmatched motifs
    results.extend(_align_to_other_entry_members(unmatched_motifs))
    # Combine and save results
    final_df = pd.concat(results).reset_index(drop=True)
    final_df.to_csv(output_path, index=False)

    return output_path


def get_res_to_motif_mapping(motifs):
    res_to_motif_id = {}
    for m in motifs:
        for r in m.get_residues():
            if r.get_str() not in res_to_motif_id:
                res_to_motif_id[r.get_str()] = m.name
            else:
                existing_motif = res_to_motif_id[r.get_str()]
                if existing_motif.startswith("HELIX"):
                    res_to_motif_id[r.get_str()] = m.name
    return res_to_motif_id


def process_pdb_id_for_unique_residues(args):
    """Process a single PDB ID to get unique residues and mapping.

    Args:
        args: Tuple containing (pdb_id, unique_motifs)
            pdb_id: PDB ID to process
            unique_motifs: List of unique motif names

    Returns:
        Tuple containing:
            - Dictionary with PDB ID and residues
            - Dictionary with PDB ID and residue to motif mapping
    """
    pdb_id, unique_motifs = args
    try:
        res_to_motif_id = {}
        motifs = get_cached_motifs(pdb_id)
        res = []
        for m in motifs:
            if m.name not in unique_motifs:
                continue
            for r in m.get_residues():
                if r.get_str() not in res:
                    res.append(r.get_str())
                if r.get_str() not in res_to_motif_id:
                    res_to_motif_id[r.get_str()] = m.name
                else:
                    existing_motif = res_to_motif_id[r.get_str()]
                    if existing_motif.startswith("HELIX"):
                        res_to_motif_id[r.get_str()] = m.name

        return ({"pdb_id": pdb_id, "residues": res}, {"pdb_id": res_to_motif_id})
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


def get_dssr_motifs_for_pdb(pdb_id: str):
    """Process a single PDB ID to get DSSR motifs.

    Args:
        pdb_id: The PDB ID to process

    Returns:
        DataFrame containing motif data or None if processing failed
    """
    try:
        pdb_data = get_pdb_structure_data(pdb_id)

    except Exception as e:
        print(f"Error getting motifs for {pdb_id}: {e}")
        return None

    path = os.path.join(DATA_PATH, "dataframes", "dssr_motifs", f"{pdb_id}.json")
    if os.path.exists(path):
        return None

    data = []
    for m in motifs:
        has_singlet_flank = False
        for b in m.basepair_ends:
            key = b.res_1.get_str() + "-" + b.res_2.get_str()
            if key in mf.singlet_pairs_lookup:
                has_singlet_flank = True
                break
        data.append(get_data_from_motif(m, pdb_id, has_singlet_flank))

    df = pd.DataFrame(data)
    df.to_json(path, orient="records")
    return df


# cli ############################################################################


@click.group()
def cli():
    pass


# Step 1: Get non-redundant motifs
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", type=int, default=1)
def get_non_redundant_motifs(csv_path, processes):
    parser = NonRedundantSetParser()
    sets = parser.parse(csv_path)
    all_args = []
    for set_id, repr_entry, child_entries in sets:
        all_args.append((set_id, repr_entry, child_entries))

    results = run_w_processes_in_batches(
        items=all_args,
        func=find_unique_motifs_in_non_redundant_set_entry,
        processes=processes,
        batch_size=100,
        desc="Processing non-redundant set entries",
    )


# Step 2: Get unique motifs
# need to run scripts/check_motifs.py to get details first
@cli.command()
def get_unique_motifs():
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets", "*.csv")
    )
    df = concat_dataframes_from_files(csv_files)
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "check_motifs", "*.csv")
    )
    df_issues = concat_dataframes_from_files(csv_files)
    df_issues = df_issues.drop(columns=["pdb_id"])
    df = df.query("is_duplicate == False").copy()
    df = df[["motif", "child_pdb"]]
    print(len(df))
    df.rename(columns={"child_pdb": "pdb_id", "motif": "motif_name"}, inplace=True)
    df = df.merge(df_issues, on="motif_name", how="left")
    path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv")
    df.to_csv(path, index=False)


# Step 3: Get unique residues
@cli.command()
@click.option("-p", "--processes", type=int, default=1)
def get_unique_residues(processes):
    """Get unique residues from non-redundant motifs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    # Read input data
    df = pd.read_csv(os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv"))
    unique_motifs = df["motif"].values
    df = add_motif_name_columns(df, "motif")
    # Prepare arguments for parallel processing
    pdb_ids = df["pdb_id"].unique()
    args = [(pdb_id, unique_motifs) for pdb_id in pdb_ids]

    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=args,
        func=process_pdb_id_for_unique_residues,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for unique residues",
    )
    # Collect and aggregate results
    data = []
    res_mapping = []
    for result in results:
        if result is not None:
            data.append(result[0])
            res_mapping.append(result[1])

    # Save results
    df = pd.DataFrame(data)
    df.to_json(
        os.path.join(DATA_PATH, "summaries", "unique_residues.json"), orient="records"
    )
    df_res_mapping = pd.DataFrame(res_mapping)
    df_res_mapping.to_json(
        os.path.join(DATA_PATH, "summaries", "res_mapping.json"), orient="records"
    )


# Step 4: Get DSSR motifs
@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def get_dssr_motifs(processes):
    """Get DSSR motifs for all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    pdb_ids = get_pdb_ids()
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=get_dssr_motifs_for_pdb,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for DSSR motifs",
    )


@cli.command()
def compare_dssr_motifs():
    comparer = MotifSetComparerer()
    pdb_ids = get_pdb_ids()
    for pdb_id in pdb_ids:
        new_path = os.path.join(
            DATA_PATH, "dataframes", "dssr_motifs_compared", f"{pdb_id}.json"
        )
        if os.path.exists(new_path):
            continue
        path = os.path.join(DATA_PATH, "dataframes", "dssr_motifs", f"{pdb_id}.json")
        dssr_motifs = pd.read_json(path)
        comparer.compare_motifs(pdb_id, dssr_motifs)
        exit()

    exit()

    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "dssr_motifs_compared", "*.json")
    )
    dfs = []
    for json_file in json_files:
        df = pd.read_json(json_file)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_json(
        os.path.join("dssr_motifs_compared.json"),
        orient="records",
    )


@cli.command()
@click.argument("csv_paths", type=click.Path(exists=True), nargs=-1)
def get_atlas_motifs_summary(csv_paths):
    # should be hl, il, j3
    dfs = []
    for csv_path in csv_paths:
        df = parse_atlas_csv(csv_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_json(
        os.path.join("atlas_motifs.json"),
        orient="records",
    )


@cli.command()
def get_atlas_motifs():
    df = pd.read_json("atlas_motifs.json")
    for pdb_id, g in df.groupby("pdb_id"):
        try:
            residues = get_cached_residues(pdb_id)
            basepairs = get_cached_basepairs(pdb_id)
            chains = Chains(get_rna_chains(residues.values()))
            other_mf = MotifFactoryFromOther(pdb_id, chains, residues, basepairs)
            mf = MotifFactory(pdb_id, chains, basepairs, [])
        except Exception as e:
            print(f"Error getting motifs for {pdb_id}: {e}")
            continue
        motifs = []
        for i, row in g.iterrows():
            motifs.append(other_mf.get_motif_from_atlas(row["mtype"], row["residues"]))
        path = os.path.join(DATA_PATH, "dataframes", "atlas_motifs", f"{pdb_id}.json")
        data = []
        for m in motifs:
            has_singlet_flank = False
            for b in m.basepair_ends:
                key = b.res_1.get_str() + "-" + b.res_2.get_str()
                if key in mf.singlet_pairs_lookup:
                    has_singlet_flank = True
                    break
            data.append(get_data_from_motif(m, pdb_id, has_singlet_flank))
        df = pd.DataFrame(data)
        df.to_json(path, orient="records")


@cli.command()
def compare_atlas_motifs():
    comparer = MotifSetComparerer()
    pdb_ids = get_pdb_ids()
    for pdb_id in pdb_ids:
        new_path = os.path.join(
            DATA_PATH, "dataframes", "atlas_motifs_compared", f"{pdb_id}.json"
        )
        if os.path.exists(new_path):
            continue
        path = os.path.join(DATA_PATH, "dataframes", "atlas_motifs", f"{pdb_id}.json")
        try:
            df = pd.read_json(path)
        except Exception as e:
            continue
        df_atlas = comparer.compare_motifs(pdb_id, df)
        df_atlas = df_atlas[df_atlas["mtype"].isin(["HAIRPIN", "TWOWAY", "NWAY"])]
        df_atlas.to_json(new_path, orient="records")


if __name__ == "__main__":
    cli()
