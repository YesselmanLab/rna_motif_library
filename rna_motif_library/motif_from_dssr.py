# Not sure going to use this stuff anymore see motif.py for newer stuff
# Standard library imports
import json
import os
from typing import Any, Dict, List, Tuple

# Third party imports
from pydssr.dssr import DSSROutput
from pydssr.dssr_classes import DSSR_PAIR

# Local imports
from rna_motif_library.basepair import Basepair
from rna_motif_library.chain import (
    are_residues_connected,
    do_strands_have_helix_sequence,
)
from rna_motif_library.hbond import Hbond
from rna_motif_library.logger import get_logger
from rna_motif_library.motif import Motif
from rna_motif_library.residue import Residue
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import wc_basepairs_w_gu

log = get_logger("motif_from_dssr")


class ChainGenerator:
    def generate_chains(self, residues: List[Residue]) -> List[List[Residue]]:
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

    def _find_residue_roots(
        self, res_list: List[Residue]
    ) -> Tuple[List[Residue], List[Residue]]:
        """
        Find the root residues that start each RNA chain.

        A root residue is one that has a 5' to 3' connection to another residue
        but no 3' to 5' connection from another residue (i.e. it's at the 5' end).

        Args:
            res_list: List of Residue objects to analyze

        Returns:
            Tuple containing:
            - List of root residues found
            - Modified list with roots removed
        """
        roots = []

        # Check each residue to see if it's a root
        for source_res in res_list:
            has_incoming = False  # 3' to 5' connection
            # Compare against all other residues
            for target_res in res_list:
                if source_res == target_res:
                    continue
                connection = are_residues_connected(source_res, target_res)
                if connection == -1:  # 3' to 5'
                    has_incoming = True
                    break  # Can stop checking once we find an incoming connection
            # Root residues have outgoing but no incoming connections
            if not has_incoming:
                roots.append(source_res)

        # Return roots and remaining residues
        remaining = [res for res in res_list if res not in roots]
        return roots, remaining

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
                    if are_residues_connected(current_residue, res) == 1:
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


def are_end_residues_chain_ends(
    residues: Dict[str, Residue], strands: List[List[Residue]]
) -> bool:
    motif_res = {}
    for strand in strands:
        for res in strand:
            motif_res[res.get_x3dna_str()] = res
    for strand in strands:
        connected_residues = [None, None]
        for res in residues.values():
            if res.get_x3dna_str() in motif_res:
                continue
            if are_residues_connected(res, strand[0]) == 1:
                connected_residues[0] = res
            elif are_residues_connected(res, strand[-1]) == -1:
                connected_residues[1] = res
        if connected_residues[0] is None or connected_residues[1] is None:
            return True
    return False


class MotifFactoryFromDSSR:
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
        self.chain_generator = ChainGenerator()
        self.wc_pairs = ["GC", "CG", "GU", "UG", "AU", "UA"]
        self.used_names = {}

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
        motifs = []
        log.info(f"Processing {len(dssr_motifs)} motifs")
        for m in dssr_motifs:
            residues = self._get_residues_for_motif(m, all_residues)
            m = self._generate_motif(residues)
            motifs.append(m)
        motifs = self._remove_strand_overlap_motifs(motifs)
        motifs = self._remove_duplicate_motifs(motifs)
        motifs = self._split_motif_into_single_strands(motifs, all_residues)
        log.info(f"Final number of motifs: {len(motifs)}")
        return motifs

    def from_residues(self, residues: List[Residue]) -> Motif:
        return self._generate_motif(residues)

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

    def _assign_end_basepairs(self, strands: List[List[Residue]]) -> List[Basepair]:
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_x3dna_str())
            end_residue_ids.append(s[-1].get_x3dna_str())

        # First collect all potential end basepairs
        end_basepairs = []
        for bp in self.basepairs:
            if (
                not bp.res_1.get_str() in end_residue_ids
                or not bp.res_2.get_str() in end_residue_ids
            ):
                continue
            # hbond score to allow for non-canonical nucelotides
            if bp.bp_type in wc_basepairs_w_gu or bp.hbond_score > 0.5:
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

    def _generate_motif(self, residues: List[Residue]) -> Motif:
        # We need to determine the data for the motif and build a class
        # First get the type
        residue_ids = [res.get_x3dna_str() for res in residues]
        strands = self.chain_generator.generate_chains(residues)
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_x3dna_str())
            end_residue_ids.append(s[-1].get_x3dna_str())
        basepairs = []
        for bp in self.basepairs:
            if bp.res_1.get_str() in residue_ids and bp.res_2.get_str() in residue_ids:
                basepairs.append(bp)
        end_basepairs = self._assign_end_basepairs(strands)
        hbonds = []
        for hb in self.hbonds:
            if hb.res_1.get_str() in residue_ids and hb.res_2.get_str() in residue_ids:
                hbonds.append(hb)
        mtype = "UNKNOWN"
        num_of_loop_strands = 0
        num_of_wc_pairs = 0
        for bp in basepairs:
            bp_name = bp.res_1.res_id + bp.res_2.res_id
            if bp.bp_type == "WC":
                num_of_wc_pairs += 1
            elif bp_name == "GU" or bp_name == "UG":
                num_of_wc_pairs += 1
        for s in strands:
            if self._is_strand_a_loop(s, end_basepairs):
                num_of_loop_strands += 1
        if len(end_basepairs) == 0 and num_of_loop_strands == 0:
            mtype = "SSTRAND"
        elif len(end_basepairs) == 1 and num_of_loop_strands == 1:
            mtype = "HAIRPIN"
        elif len(end_basepairs) == 2 and len(strands) == 2 and num_of_loop_strands == 0:
            if do_strands_have_helix_sequence(strands):
                mtype = "HELIX"
            else:
                mtype = "TWOWAY"
        elif len(end_basepairs) > 2 and num_of_loop_strands == 0:
            mtype = "NWAY"
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
        mname = f"{self.pdb_name}-{mtype}-{sequence}"
        if mname in self.used_names:
            self.used_names[mname] += 1
        else:
            self.used_names[mname] = 1
        mname += f"-{self.used_names[mname]}"
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

    def _remove_duplicate_motifs(self, motifs: List[Any]) -> List[Any]:
        unique_motifs = []
        unique_motif_residues = {}
        for i, motif in enumerate(motifs):
            duplicate = False
            for j, unique_motif in enumerate(unique_motifs):
                if i <= j:
                    continue
                if motif.sequence != unique_motif.sequence:
                    continue
                contains_residue = True
                for res in motif.get_residues():
                    if (
                        res.get_x3dna_str()
                        not in unique_motif_residues[unique_motif.name]
                    ):
                        contains_residue = False
                        break
                if contains_residue:
                    duplicate = True
                    break
            if not duplicate:
                unique_motifs.append(motif)
                unique_motif_residues[motif.name] = []
                for res in motif.get_residues():
                    unique_motif_residues[motif.name].append(res.get_x3dna_str())
        return unique_motifs

    # TODO come back to this after other processing
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

    def _extract_hairpin_motif(self, hairpin_motif, other_motif):
        # Find the hairpin strand in the other motif
        hairpin_strand = None
        for strand in other_motif.strands:
            # Check if this strand is contained within any of the hairpin motif's strands
            for hairpin_strand_candidate in hairpin_motif.strands:
                if all(res in hairpin_strand_candidate for res in strand):
                    hairpin_strand = strand
                    break
            if hairpin_strand is not None:
                break

        residues = []
        for strand in other_motif.strands:
            if strand != hairpin_strand:
                residues.extend(strand)

        new_motif = self.from_residues(residues)
        return new_motif

    def _remove_strand_overlap_motifs(self, motifs: List[Motif]) -> List[Motif]:
        """
        DSSR sometimes thinks hairpin strands are contained in larger motifs but are
        actually tertiary contacts and need to be seperated.
        """
        contained_motifs = []
        new_motifs = []
        for i, motif1 in enumerate(motifs):
            found = False
            for j, motif2 in enumerate(motifs):
                if i >= j:
                    continue
                for strand1 in motif1.strands:
                    for strand2 in motif2.strands:
                        if all(res in strand1 for res in strand2):
                            contained_motifs.append((motif1, motif2))
                            found = True
                            break
                        else:
                            continue
            if not found:
                new_motifs.append(motif1)

        for motif1, motif2 in contained_motifs:
            if not ("HAIRPIN" in motif1.name or "HAIRPIN" in motif2.name):
                continue
            if "HAIRPIN" in motif1.name and "HAIRPIN" in motif2.name:
                if len(motif1.get_residues()) < len(motif2.get_residues()):
                    hairpin_motif = motif1
                    other_motif = motif2
                else:
                    hairpin_motif = motif2
                    other_motif = motif1
            elif "HAIRPIN" in motif1.name:
                hairpin_motif = motif1
                other_motif = motif2
            else:
                hairpin_motif = motif2
                other_motif = motif1
            log.info(
                f"Extracting hairpin motif {hairpin_motif.name} from {other_motif.name}"
            )
            new_motif = self._extract_hairpin_motif(hairpin_motif, other_motif)
            new_motifs.append(new_motif)
            new_motifs.append(hairpin_motif)
        return new_motifs

    def _split_motif_into_single_strands(
        self, motifs: List[Motif], residues: Dict[str, Residue]
    ) -> List[Motif]:
        new_motifs = []
        for motif in motifs:
            if motif.mtype != "UNKNOWN":
                new_motifs.append(motif)
                continue
            if not are_end_residues_chain_ends(residues, motif.strands):
                new_motifs.append(motif)
                continue
            log.debug(f"Splitting motif {motif.name} into single strands")
            for strand in motif.strands:
                res = strand
                if motif.residue_has_basepair(res[0]):
                    if len(res) > 1:
                        res = res[1:]
                    else:
                        continue
                if motif.residue_has_basepair(res[-1]):
                    if len(res) > 1:
                        res = res[:-1]
                    else:
                        continue
                new_motifs.append(self.from_residues(res))
        return new_motifs
