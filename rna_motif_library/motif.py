import json
import os
from dataclasses import dataclass
from typing import List, Any, Tuple, Union, Dict
import pandas as pd
import numpy as np

from pydssr.dssr import DSSROutput
from pydssr.dssr_classes import DSSR_PAIR

from rna_motif_library.classes import (
    extract_longest_numeric_sequence,
    sanitize_x3dna_atom_name,
    X3DNAResidue,
    Residue,
    Hbond,
    Basepair,
)

from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.interactions import get_hbonds_and_basepairs
from rna_motif_library.logger import get_logger

log = get_logger("motif")


def get_motifs(pdb_name: str) -> list:
    """Process motifs and interactions from a PDB file"""
    hbonds, basepairs = get_hbonds_and_basepairs(pdb_name)

    mp = MotifFactory(pdb_name, hbonds, basepairs)
    return mp.process()


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

    @classmethod
    def from_dict(cls, d: dict):
        # Convert nested objects back to their proper classes
        strands = [[Residue.from_dict(r) for r in strand] for strand in d["strands"]]
        basepairs = [Basepair.from_dict(bp) for bp in d["basepairs"]]
        basepair_ends = [Basepair.from_dict(bp) for bp in d["basepair_ends"]]
        hbonds = [Hbond.from_dict(hb) for hb in d["hbonds"]]

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
        )

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

    def contains_residue(self, residue: Residue) -> bool:
        for strand in self.strands:
            for res in strand:
                if res.get_x3dna_str() == residue.get_x3dna_str():
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

    def to_dict(self):
        return {
            "name": self.name,
            "mtype": self.mtype,
            "pdb": self.pdb,
            "size": self.size,
            "sequence": self.sequence,
            "strands": [res.to_dict() for res in self.get_residues()],
            "basepairs": [bp.to_dict() for bp in self.basepairs],
            "basepair_ends": [bp.to_dict() for bp in self.basepair_ends],
            "hbonds": [hb.to_dict() for hb in self.hbonds],
        }


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


class MotifFactory:
    """Class for processing motifs and interactions from PDB files"""

    def __init__(self, pdb_name: str, hbonds: List[Hbond], basepairs: List[Basepair]):
        """
        Initialize the MotifProcessor

        Args:
            count (int): # of PDBs processed (loaded from outside)
            pdb_path (str): path to the source PDB
        """
        self.pdb_name = pdb_name
        self.hbonds = hbonds
        self.basepairs = basepairs

    def process(self) -> List[Motif]:
        """
        Process the PDB file and extract motifs and interactions

        Returns:
            motif_list (list): list of motif names
        """
        log.debug(f"{self.pdb_name}")

        residue_data = json.loads(
            open(
                os.path.join(DATA_PATH, "jsons", "residues", f"{self.pdb_name}.json")
            ).read()
        )
        all_residues = {k: Residue.from_dict(v) for k, v in residue_data.items()}
        json_path = os.path.join(DATA_PATH, "dssr_output", f"{self.pdb_name}.json")
        dssr_output = DSSROutput(json_path=json_path)
        dssr_motifs = dssr_output.get_motifs()
        dssr_tertiary_contacts = dssr_output.get_tertiary_contacts()
        # TODO check other types of DSSR classes like kissing loops
        for t in dssr_tertiary_contacts:
            print(t)
        # Process each motif
        motifs = []
        log.info(f"Processing {len(dssr_motifs)} motifs")
        for m in dssr_motifs:
            mi = self._generate_motif(m, all_residues)
            motifs.append(mi)
        return motifs
        """for i, motif1 in enumerate(motifs):
            for j, motif2 in enumerate(motifs):
                if i >= j:
                    continue
                # Count residue overlap between motifs
                overlap_count = 0
                for res1 in motif1.get_residues():
                    for res2 in motif2.get_residues():
                        if res1 == res2:
                            overlap_count += 1
                if overlap_count > 2:
                    print(motif1.name, motif2.name, overlap_count)
        """
        contained_motifs = []
        for i, motif1 in enumerate(motifs):
            for j, motif2 in enumerate(motifs):
                if i >= j:
                    continue
                for strand1 in motif1.strands:
                    for strand2 in motif2.strands:
                        if all(res in strand1 for res in strand2):

                            contained_motifs.append((motif1, motif2))
                            break
                    else:
                        continue

        return motifs

    def _get_residues_for_motif(
        self, m: Any, all_residues: Dict[str, Residue]
    ) -> List[Residue]:
        residues = []
        for nt in m.nts_long:
            if nt in all_residues:
                residues.append(all_residues[nt])
            else:
                log.warning(f"Residue {nt} not found in {self.pdb_name}")
                continue
        return residues

    def _is_strand_a_loop(
        self, strand: List[Residue], end_basepairs: List[Basepair]
    ) -> bool:
        end_residue_ids = []
        end_residue_ids.append(strand[0].get_x3dna_str())
        end_residue_ids.append(strand[-1].get_x3dna_str())
        for bp in end_basepairs:
            if (
                bp.res_1.get_str() in end_residue_ids
                and bp.res_2.get_str() in end_residue_ids
            ):
                return True
        return False

    def _generate_motif(self, m: Any, all_residues: Dict[str, Residue]) -> Motif:
        # We need to determine the data for the motif and build a class
        # First get the type
        residues = self._get_residues_for_motif(m, all_residues)
        residue_ids = [res.get_x3dna_str() for res in residues]
        strands = self._generate_strands(residues)
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_x3dna_str())
            end_residue_ids.append(s[-1].get_x3dna_str())
        basepairs = []
        for bp in self.basepairs:
            if bp.res_1.get_str() in residue_ids and bp.res_2.get_str() in residue_ids:
                basepairs.append(bp)
        end_basepairs = []
        for bp in basepairs:
            if (
                bp.res_1.get_str() in end_residue_ids
                and bp.res_2.get_str() in end_residue_ids
            ):
                end_basepairs.append(bp)
        hbonds = []
        for hb in self.hbonds:
            if hb.res_1.get_str() in residue_ids and hb.res_2.get_str() in residue_ids:
                hbonds.append(hb)
        mtype = "UNKNOWN"
        num_of_loop_strands = 0
        num_of_wc_pairs = 0
        for bp in basepairs:
            if bp.bp_type == "WC":
                num_of_wc_pairs += 1
        for s in strands:
            if self._is_strand_a_loop(s, end_basepairs):
                num_of_loop_strands += 1
        if len(end_basepairs) == 0 and num_of_loop_strands == 0:
            mtype = "SINGLE-STRAND"
        elif len(end_basepairs) == 1 and num_of_loop_strands == 1:
            mtype = "HAIRPIN"
        elif len(end_basepairs) == 2 and len(strands) == 2 and num_of_loop_strands == 0:
            if num_of_wc_pairs == len(strands[0]):
                mtype = "HELIX"
            else:
                mtype = "TWOWAY-JUNCTION"
        elif len(end_basepairs) > 2 and len(strands) == len(end_basepairs):
            mtype = "NWAY-JUNCTION"
        else:
            log.debug(
                f"Unknown motif type in {self.pdb_name} with "
                f"{len(end_basepairs)} end basepairs and "
                f"{len(strands)} strands and "
                f"{num_of_loop_strands} loop strands and "
                f"{num_of_wc_pairs} wc pairs"
            )
            res_str = ""
            for s in strands:
                res_str += " ".join([res.get_x3dna_str() for res in s]) + " "
            log.debug(res_str)
        sequence = self._find_sequence(strands).replace("&", "-")
        mname = f"{mtype}-{sequence}"
        return Motif(
            mname,
            mtype,
            self.pdb_name,
            "SIZE",
            sequence,
            strands,
            basepairs,
            end_basepairs,
            hbonds,
        )

    def _generate_strands(self, residues: List[Residue]) -> List[List[Residue]]:
        """
        Generates ordered strands of RNA residues by finding root residues and building 5' to 3'.

        Args:
            residues (List[ResidueNew]): List of RNA residues to analyze

        Returns:
            List[List[ResidueNew]]: List of RNA strands, where each strand is a list of residues ordered 5' to 3'
        """
        residue_roots, res_list_modified = self._find_residue_roots(residues)
        strands_of_rna = self._build_strands_5to3(residue_roots, res_list_modified)

        return strands_of_rna

    def _find_sequence(self, strands_of_rna: List[List[Residue]]) -> str:
        """Find sequences from found strands of RNA"""
        res_strands = []
        for strand in strands_of_rna:
            res_strand = []
            for residue in strand:
                mol_name = residue.res_id
                res_strand.append(mol_name)
            strand_sequence = "".join(res_strand)
            res_strands.append(strand_sequence)
        sequence = "&".join(res_strands)
        return sequence

    def _find_residue_roots(
        self, res_list: List[Residue]
    ) -> Tuple[List[Residue], List[Residue]]:
        """
        Find the root residues that start each RNA chain.

        A root residue is one that has a 5' to 3' connection to another residue
        but no 3' to 5' connection from another residue (i.e. it's at the 5' end).

        Args:
            res_list: List of ResidueNew objects to analyze

        Returns:
            Tuple containing:
            - List of root residues found
            - Modified list with roots removed
        """
        roots = []

        # Check each residue to see if it's a root
        for source_res in res_list:
            has_outgoing = False  # 5' to 3' connection
            has_incoming = False  # 3' to 5' connection

            # Compare against all other residues
            for target_res in res_list:
                if source_res == target_res:
                    continue

                connection = self._are_residues_connected(source_res, target_res)

                if connection == 1:  # 5' to 3'
                    has_outgoing = True
                elif connection == -1:  # 3' to 5'
                    has_incoming = True
                    break  # Can stop checking once we find an incoming connection
            # Root residues have outgoing but no incoming connections
            if has_outgoing and not has_incoming:
                roots.append(source_res)

        # Return roots and remaining residues
        remaining = [res for res in res_list if res not in roots]
        return roots, remaining

    def _are_residues_connected(
        self,
        source_residue: Residue,
        residue_in_question: Residue,
        cutoff: float = 2.75,
    ) -> int:
        """Determine if another residue is connected to this residue"""
        # Get O3' coordinates from source residue
        o3_coords_1 = source_residue.get_atom_coords("O3'")
        p_coords_2 = residue_in_question.get_atom_coords("P")

        # Check 5' to 3' connection
        if o3_coords_1 is not None and p_coords_2 is not None:
            distance = np.linalg.norm(np.array(p_coords_2) - np.array(o3_coords_1))
            if distance < cutoff:
                return 1

        # Check 3' to 5' connection
        o3_coords_2 = residue_in_question.get_atom_coords("O3'")
        p_coords_1 = source_residue.get_atom_coords("P")

        if o3_coords_2 is not None and p_coords_1 is not None:
            distance = np.linalg.norm(np.array(o3_coords_2) - np.array(p_coords_1))
            if distance < cutoff:
                return -1

        return 0

    def _build_strands_5to3(
        self, residue_roots: List[Residue], res_list: List[Residue]
    ):
        """Build strands of RNA from the list of given residues"""
        built_strands = []

        for root in residue_roots:
            current_residue = root
            chain = [current_residue]

            while True:
                next_residue = None
                for res in res_list:
                    if self._are_residues_connected(current_residue, res) == 1:
                        next_residue = res
                        break

                if next_residue:
                    chain.append(next_residue)
                    current_residue = next_residue
                    res_list.remove(next_residue)
                else:
                    break

            built_strands.append(chain)

        return built_strands

    # TODO come back to this after other processing
    def _remove_duplicate_motifs(self, motifs: List[Any]) -> List[Any]:
        """Remove duplicate motifs from a list of motifs"""
        duplicates = []
        for m1 in motifs:
            if m1 in duplicates:
                continue

            m1_nts = [nt.split(".")[1] for nt in m1.nts_long]

            for m2 in motifs:
                if m1 == m2:
                    continue

                m2_nts = [nt.split(".")[1] for nt in m2.nts_long]

                if m1_nts == m2_nts:
                    duplicates.append(m2)

        unique_motifs = [m for m in motifs if m not in duplicates]
        return unique_motifs

    def _merge_singlet_separated(self, motifs: List[Motif]) -> List[Motif]:
        target_motifs = []
        for mi in motifs:
            if len(mi.basepair_ends) > 2:
                target_motifs.append(mi)
        merged = []
        removed = []
        for i, mi in enumerate(target_motifs):
            for j, mj in enumerate(target_motifs):
                if i == j:
                    continue
                shared_bp = False
                for bp1 in mi.basepair_ends:
                    for bp2 in mj.basepair_ends:
                        if bp1 == bp2:
                            shared_bp = True
                            break
                if shared_bp:
                    print(mi.mname, mj.mname)

        return []
