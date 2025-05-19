from rna_motif_library.motif import Motif, save_motifs_to_json
from rna_motif_library.chain import (
    Chains,
    get_rna_chains,
    get_cached_chains,
    write_chain_to_cif,
)
from rna_motif_library.basepair import Basepair, get_cached_basepairs
from rna_motif_library.hbond import Hbond, get_cached_hbonds
from rna_motif_library.residue import Residue
from rna_motif_library.logger import get_logger
from rna_motif_library.settings import RESOURCES_PATH
from typing import List, Dict, Set, Tuple, Optional
import os
from dataclasses import dataclass


log = get_logger(__name__)


@dataclass
class PDBStructureData:
    pdb_id: str
    chains: Chains
    basepairs: List[Basepair]
    hbonds: List[Hbond]


def get_pdb_structure_data(pdb_id: str) -> PDBStructureData:
    chains = Chains(get_cached_chains(pdb_id))
    basepairs = get_cached_basepairs(pdb_id)
    hbonds = get_cached_hbonds(pdb_id)
    return PDBStructureData(pdb_id, chains, basepairs, hbonds)


def get_pdb_structure_data_for_residues(
    pdb_data: PDBStructureData, residues: List[Residue]
) -> PDBStructureData:
    chains = Chains(get_rna_chains(residues))
    basepairs = get_basepairs_for_residue(residues, pdb_data.basepairs)
    return PDBStructureData(pdb_data.pdb_id, chains, basepairs, pdb_data.hbonds)


# reusable functions ################################################################


def get_valid_pairs() -> List[str]:
    """Get list of valid cWW basepair types from file.

    Returns:
        List of valid basepair type strings (e.g. ['A-U', 'G-C', etc])
    """
    f = open(os.path.join(RESOURCES_PATH, "valid_cww_pairs.txt"))
    lines = f.readlines()
    allowed_pairs = []
    for line in lines:
        allowed_pairs.append(line.strip())
    f.close()
    return allowed_pairs


def get_cww_basepairs(
    pdb_data: PDBStructureData,
    valid_pairs: Optional[List[str]] = None,
    min_two_hbond_score: float = 1.3,
    min_three_hbond_score: float = 2.0,
) -> Dict[str, Basepair]:
    """Get dictionary of cWW (cis Watson-Watson) basepairs from a PDB structure.

    Args:
        pdb_data: PDBStructureData object containing basepairs and chains information
        valid_pairs: Optional list of valid basepair types (e.g. ['A-U', 'G-C']). If None, loads from file.
        min_two_hbond_score: Minimum required hbond score for A-U and G-U pairs (default 1.3)
        min_three_hbond_score: Minimum required hbond score for G-C pairs (default 2.0)

    Returns:
        Dictionary mapping residue pair strings (e.g. "A1-U20") to Basepair objects.
        Each basepair is stored twice with keys in both directions (A1-U20 and U20-A1).

    Note:
        Only includes basepairs that:
        - Have cWW geometry
        - Are in the valid_pairs list
        - Have both residues present in the structure
        - Meet minimum hbond score thresholds based on pair type
    """
    basepairs = pdb_data.basepairs
    chains = pdb_data.chains
    try:
        if valid_pairs is None:
            valid_pairs = get_valid_pairs()
    except FileNotFoundError:
        log.error(
            f"Valid pairs file not found: {os.path.join(RESOURCES_PATH, 'valid_cww_pairs.txt')}"
        )
        valid_pairs = None
    cww_basepairs = {}
    two_hbond_pairs = ["A-U", "U-A", "G-U", "U-G"]
    three_hbond_pairs = ["G-C", "C-G"]
    for bp in basepairs:
        if bp.lw != "cWW" or bp.bp_type not in valid_pairs:
            continue
        if (
            chains.get_residue_by_str(bp.res_1.get_str()) is None
            or chains.get_residue_by_str(bp.res_2.get_str()) is None
        ):
            continue
        # stops a lot of bad basepairs from being included
        if bp.bp_type in two_hbond_pairs and bp.hbond_score < min_two_hbond_score:
            continue
        if bp.bp_type in three_hbond_pairs and bp.hbond_score < min_three_hbond_score:
            continue
        key1 = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
        key2 = f"{bp.res_2.get_str()}-{bp.res_1.get_str()}"
        cww_basepairs[key1] = bp
        cww_basepairs[key2] = bp
    return cww_basepairs


def get_basepairs_for_strands(
    strands: List[List[Residue]],
    basepairs: List[Basepair],
    chains: Chains,
) -> List[Basepair]:
    basepairs = []
    residues = []
    for strand in strands:
        for res in strand:
            residues.append(res.get_str())
    for bp in basepairs:
        res1, res2 = chains.get_residues_in_basepair(bp)
        if res1 is None or res2 is None:
            continue
        if res1.get_str() in residues and res2.get_str() in residues:
            basepairs.append(bp)
    return basepairs


def get_basepairs_for_residue(
    residues: List[Residue], basepairs: List[Basepair]
) -> List[Basepair]:
    res_basepairs = []
    residue_strs = [r.get_str() for r in residues]
    for bp in basepairs:
        if bp.res_1.get_str() in residue_strs or bp.res_2.get_str() in residue_strs:
            res_basepairs.append(bp)
    return res_basepairs


def get_basepair_ends_for_strands(
    strands: List[List[Residue]], basepairs: List[Basepair]
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


class HelixFinder:
    def __init__(
        self,
        pdb_data: PDBStructureData,
        cww_basepairs: Dict[str, Basepair],
        hairpins: List[Motif],
    ):
        self.chains = pdb_data.chains
        self.cww_basepairs = cww_basepairs
        self.hairpins = hairpins
        self.count = -1

    def get_helices(self) -> List[Motif]:
        """Find all possible helices in the structure."""
        # Get residues already used in hairpins
        # this avoids confusing pseudoknots with helices
        hairpin_res_strs = []
        for h in self.hairpins:
            # exclude first and last residue which are a cWW basepair
            for res in h.strands[0][1:-1]:
                hairpin_res_strs.append(res.get_str())
        # Filter to only valid cWW basepairs not in hairpins
        cww_basepairs = []
        for bp in self.cww_basepairs.values():
            if (
                bp.res_1.get_str() in hairpin_res_strs
                or bp.res_2.get_str() in hairpin_res_strs
            ):
                continue
            cww_basepairs.append(bp)

        # First find all possible helices
        potential_helices = []
        for bp in cww_basepairs:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            helix = self._build_helix(bp, hairpin_res_strs)
            if helix:
                potential_helices.append(helix)

        # Filter out shorter helices that share residues with longer ones
        final_helices = []
        potential_helices.sort(key=lambda x: len(x.basepairs), reverse=True)
        used_residues = set()

        for helix in potential_helices:
            helix_residues = set()
            for res in helix.get_residues():
                helix_residues.add(res.get_str())

            # Check if this helix shares any residues with longer ones
            if not helix_residues.intersection(used_residues):
                final_helices.append(helix)
                used_residues.update(helix_residues)

        return final_helices

    def _is_helix_start(
        self, bp: Basepair, cww_basepairs: Dict[str, Basepair], used_basepairs: Set[str]
    ) -> bool:
        """Check if basepair could start a new helix."""
        prev_pair = self.chains.get_prev_pair_id(bp)
        if prev_pair is None:
            return True
        if prev_pair in cww_basepairs:
            return False
        else:
            return True

    def _build_helix(
        self,
        start_bp: Basepair,
        hairpin_res_strs: List[str],
    ) -> Optional[Motif]:
        """Build a helix starting from the given basepair."""
        current_bp = start_bp

        # Follow chain of basepairs
        used_basepairs = set()
        seen = []
        open_bp = [start_bp]
        helix = []

        while open_bp:
            current_bp = open_bp.pop(0)
            helix.append(current_bp)
            flanking_bps = self.chains.get_possible_flanking_bps(current_bp)
            for res1, res2 in flanking_bps:
                if res1 is None or res2 is None:
                    continue
                key1 = f"{res1.get_str()}-{res2.get_str()}"
                key2 = f"{res2.get_str()}-{res1.get_str()}"
                if key1 not in self.cww_basepairs:
                    continue
                if key1 in used_basepairs or key1 in seen:
                    continue
                if key2 in used_basepairs or key2 in seen:
                    continue
                if (
                    res1.get_str() in hairpin_res_strs
                    or res2.get_str() in hairpin_res_strs
                ):
                    continue
                bp = self.cww_basepairs[key1]
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

        self.count += 1
        return Motif(
            f"HELIX-{self.count}",
            "HELIX",
            "",
            "",
            "",
            strands,
            helix,
            get_basepair_ends_for_strands(strands, helix),
            [],
        )


class HairpinFinder:
    def __init__(self, pdb_data: PDBStructureData, cww_basepairs: Dict[str, Basepair]):
        self.basepairs = pdb_data.basepairs
        self.chains = pdb_data.chains
        self.cww_basepairs = cww_basepairs

    def get_hairpins(self) -> List[Motif]:
        """Find all possible hairpins in the structure."""
        bps = self._get_potential_hairpin_basepairs()
        distances = self._get_hairpin_distances(bps)
        hairpins = self._get_shorest_hairpin_motifs(distances)
        return hairpins

    def _get_potential_hairpin_basepairs(self) -> Dict[str, int]:
        """Find basepairs that could potentially form hairpins.

        Filters basepairs to only include those where:
        - Both residues are on the same chain
        - Adjacent residues are not basepaired
        - Previous residues are basepaired (i.e. are singlet)
        Returns:
            List of Basepair objects that could form hairpins
        """
        bps = []
        for bp in self.cww_basepairs.values():
            res1, res2 = self.chains.get_residues_in_basepair(bp)
            if not self.chains.are_residues_on_same_chain(res1, res2):
                continue
            if not self._are_next_residues_unpaired(res1, res2):
                continue
            if self._are_previous_residues_unpaired(res1, res2):
                continue
            bps.append(bp)
        return bps

    def _get_hairpin_distances(self, bps: List[Basepair]) -> Dict[str, int]:
        """Calculate distances between basepaired residues that could form hairpins.

        For each basepair, calculates the length of the chain of residues between
        the two paired residues. This represents the size of the potential hairpin loop.

        Returns:
            Dict mapping basepair strings ('res1-res2') to the number of residues
            between the paired residues.
        """
        distances = {}
        for bp in bps:
            chain = self.chains.get_chain_between_basepair(bp)
            distances[f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"] = len(chain)
        return distances

    def _get_shorest_hairpin_motifs(self, distances: Dict[str, int]) -> List[Motif]:
        """Build Motif objects for valid hairpins."""
        sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
        hairpins = []
        used_residues = set()
        for pos, (pair_str, _) in enumerate(sorted_pairs):
            bp = self.cww_basepairs[pair_str]
            chain = self.chains.get_chain_between_basepair(bp)
            # Skip if overlaps with existing hairpin
            if any(res.get_str() in used_residues for res in chain):
                continue
            # Add residues to used set
            used_residues.update(res.get_str() for res in chain)
            basepairs = get_basepairs_for_strands([chain], self.basepairs, self.chains)

            sequence = "".join([r.res_id for r in chain])
            hairpins.append(
                Motif(
                    f"HAIRPIN-{pos}",
                    "HAIRPIN",
                    "",
                    "",
                    sequence,
                    [chain],
                    basepairs,
                    [bp],
                    [],
                )
            )

        return hairpins

    def _are_next_residues_unpaired(
        self,
        res1: Residue,
        res2: Residue,
    ) -> bool:
        """Check if residues adjacent to a basepair are not themselves basepaired.

        For a given basepair (res1-res2), checks if the residues immediately adjacent
        to them (res1+1 and res2-1) form a cWW basepair. This helps identify valid
        hairpin loops which should not have consecutive basepairs.

        Args:
            res1: First residue in the basepair
            res2: Second residue in the basepair

        Returns:
            True if the adjacent residues are not basepaired, False if they are
        """
        chain_num_1, pos_1 = self.chains.get_residue_position(res1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res2)

        if pos_1 > pos_2:
            pos_1, pos_2 = pos_2, pos_1

        next_res_1 = self.chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
        next_res_2 = self.chains.get_residue_by_pos(chain_num_2, pos_2 - 1)

        # Check if adjacent residues are also basepaired
        key = f"{next_res_1.get_str()}-{next_res_2.get_str()}"
        return key not in self.cww_basepairs

    def _are_previous_residues_unpaired(
        self,
        res1: Residue,
        res2: Residue,
    ) -> bool:
        """Check if residues adjacent to a basepair are not themselves basepaired."""
        chain_num_1, pos_1 = self.chains.get_residue_position(res1)
        chain_num_2, pos_2 = self.chains.get_residue_position(res2)

        if pos_1 > pos_2:
            pos_1, pos_2 = pos_2, pos_1

        prev_res_1 = self.chains.get_residue_by_pos(chain_num_1, pos_1 - 1)
        prev_res_2 = self.chains.get_residue_by_pos(chain_num_2, pos_2 + 1)
        if prev_res_1 is None or prev_res_2 is None:
            return False

        key = f"{prev_res_1.get_str()}-{prev_res_2.get_str()}"
        return key not in self.cww_basepairs


# MotifFactory #######################################################################


class MotifFactory:
    def __init__(
        self,
        pdb_data: PDBStructureData,
    ):
        self.pdb_data = pdb_data
        self.pdb_id = pdb_data.pdb_id
        self.chains = pdb_data.chains
        self.basepairs = pdb_data.basepairs
        self.hbonds = pdb_data.hbonds
        self.motif_name_count = {}
        # self.cww_basepairs_lookup = get_cww_basepairs(pdb_data)
        self.cww_residues_to_basepairs = {}
        self.cww_basepairs_lookup = get_cww_basepairs(
            pdb_data, min_two_hbond_score=0.5, min_three_hbond_score=0.5
        )
        self.motif_count = -1
        # Group basepairs by residue
        for bp in self.cww_basepairs_lookup.values():
            if bp.res_1.get_str() not in self.cww_residues_to_basepairs:
                self.cww_residues_to_basepairs[bp.res_1.get_str()] = []
            if bp.res_2.get_str() not in self.cww_residues_to_basepairs:
                self.cww_residues_to_basepairs[bp.res_2.get_str()] = []
            self.cww_residues_to_basepairs[bp.res_1.get_str()].append(bp)
            self.cww_residues_to_basepairs[bp.res_2.get_str()].append(bp)

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
        hpf = HairpinFinder(self.pdb_data, self.cww_basepairs_lookup)
        possible_hairpins = hpf.get_hairpins()
        hf = HelixFinder(self.pdb_data, self.cww_basepairs_lookup, possible_hairpins)
        helices = hf.get_helices()
        non_helical_strands = self.get_non_helical_strands(helices)
        strands_between_helices = self.get_strands_between_helices(helices)
        potential_motifs = self.get_non_canonical_motifs(
            non_helical_strands + strands_between_helices,
            self.cww_basepairs_lookup.values(),
        )
        # potential_motifs, helices = self.check_for_poor_quality_motifs(
        #    potential_motifs, helices
        # )
        finished_motifs, unfinished_motifs = self._find_and_assign_finished_motifs(
            potential_motifs
        )
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

    def check_for_poor_quality_motifs(
        self, current_motifs: List[Motif], current_helices: List[Motif]
    ):
        embedded_helices = []
        new_non_helical_strands = []
        complete_motifs = []
        for m in current_motifs:
            pdb_data_for_residues = get_pdb_structure_data_for_residues(
                self.pdb_data, m.get_residues()
            )
            hf = HelixFinder(pdb_data_for_residues, self.cww_basepairs_lookup_poor, [])
            helices = hf.get_helices()
            if len(helices) == 0:
                complete_motifs.append(m)
                continue
            embedded_helices.extend(helices)
            non_helical_strands = self._get_non_helical_strands_from_residues(
                m.get_residues(), helices
            )
            new_non_helical_strands.extend(non_helical_strands)
        new_helices = self._combine_helices(embedded_helices + current_helices)
        strands_between_helices = self.get_strands_between_helices(new_helices)
        new_non_canon_motifs = self.get_non_canonical_motifs(
            new_non_helical_strands + strands_between_helices,
            self.cww_basepairs_lookup_poor.values(),
        )
        # for i, m in enumerate(new_non_canon_motifs):
        #    m.to_cif(f"motif_{i}.cif")
        return new_non_canon_motifs + complete_motifs, new_helices

    def _combine_helices(self, helices: List[Motif]) -> List[Motif]:
        final_helices = []
        # Sort helices by number of residues (largest first)
        combined = True
        while combined:
            combined = False
            while helices:
                current_helix = helices.pop(0)
                current_res = [r.get_str() for r in current_helix.get_residues()]
                overlapping_helices = []
                for h in helices:
                    overlap = False
                    h_res = [r.get_str() for r in h.get_residues()]
                    for res in h_res:
                        if res in current_res:
                            overlap = True
                            break
                    if overlap:
                        overlapping_helices.append(h)
                for h in overlapping_helices:
                    helices.remove(h)
                if len(overlapping_helices) == 0:
                    final_helices.append(current_helix)
                    continue
                all_res = []
                all_res.extend(current_helix.get_residues())
                for h in overlapping_helices:
                    for res in h.get_residues():
                        if res not in all_res:
                            all_res.append(res)
                strands = get_rna_chains(all_res)
                combined_helix = self._generate_initial_motif_from_strands(
                    strands, "HELIX", self.cww_basepairs_lookup.values()
                )
                final_helices.append(combined_helix)
                combined = True

            if combined:
                helices = final_helices[:]
                helices.sort(key=lambda h: len(h.get_residues()), reverse=True)
                final_helices = []
        return final_helices

    def _get_non_helical_strands_from_residues(
        self, residues: List[Residue], helices: List[Motif]
    ) -> List[List[Residue]]:
        non_helical_res = []
        helical_residues = []
        for h in helices:
            for res in h.get_residues():
                helical_residues.append(res.get_str())
        for res in residues:
            if res.get_str() not in helical_residues:
                non_helical_res.append(res)
        strands = get_rna_chains(non_helical_res)
        strands = self._extend_strands_with_basepairs(strands)
        return strands

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
        cww_basepairs,
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
                    unprocessed_strands[0], count, unprocessed_strands, cww_basepairs
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

    def _number_of_flanking_bps(self, bp: Basepair) -> int:
        combos = self.chains.get_possible_flanking_bps(bp)
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
        for bp in self.cww_basepairs_lookup.values():
            count += 1
            if self._number_of_flanking_bps(bp) == 0:
                singlet_pairs.append(bp)

        return singlet_pairs

    def _get_potential_basepair_ends_for_strands(
        self, strands: List[List[Residue]]
    ) -> List[Basepair]:
        basepairs = []
        for strand in strands:
            for res in strand:
                if res.get_str() in self.cww_residues_to_basepairs:
                    basepairs.extend(self.cww_residues_to_basepairs[res.get_str()])

        return basepairs

    # helix functions ##################################################################

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
            for res in h.get_residues():
                helical_residues.append(res.get_str())

        strands = []
        for chain in self.chains.chains:
            current_strand = []
            for res in chain:
                if res.get_str() not in helical_residues:
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
                            for bp in self.cww_residues_to_basepairs[res.get_str()]:
                                if bp not in current_basepair_ends:
                                    current_basepair_ends.append(bp)
                    changed = True

        return current_motif_strands, current_basepair_ends

    def _get_non_canonical_motif_from_strands(
        self,
        initial_strand: List[Residue],
        count: int,
        unprocessed_strands: List[List[Residue]],
        cww_basepairs,
    ) -> List[Motif]:
        current_motif_strands = [initial_strand]
        current_basepair_ends = []
        for res in [initial_strand[0], initial_strand[-1]]:
            if res.get_str() in self.cww_residues_to_basepairs:
                for bp in self.cww_residues_to_basepairs[res.get_str()]:
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
                            get_basepairs_for_strands(
                                [strand], self.basepairs, self.chains
                            ),
                            get_basepair_ends_for_strands([strand], cww_basepairs),
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
                get_basepairs_for_strands(
                    current_motif_strands, self.basepairs, self.chains
                ),
                get_basepair_ends_for_strands(current_motif_strands, cww_basepairs),
                [],
            )
        ]

    # finalizing motifs ################################################################
    def _generate_initial_motif_from_strands(
        self,
        strands: List[List[Residue]],
        mtype: str,
        cww_basepairs: Dict[str, Basepair],
    ) -> Motif:
        self.motif_count += 1
        return Motif(
            f"{mtype}-{self.motif_count}",
            mtype,
            "",
            "",
            "",
            strands,
            get_basepairs_for_strands(strands, self.basepairs, self.chains),
            get_basepair_ends_for_strands(strands, cww_basepairs),
            [],
        )

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
        motif.basepairs = get_basepairs_for_strands(
            motif.strands, self.basepairs, self.chains
        )
        motif.end_basepairs = get_basepair_ends_for_strands(
            motif.strands, self.cww_basepairs_lookup.values()
        )
        motif.size = self._get_motif_topology(motif)
        motif.name = self._name_motif(motif)
        res_strs = [res.get_str() for res in motif.get_residues()]
        for hbond in self.pdb_data.hbonds:
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
                    for bp in self.cww_residues_to_basepairs[res.get_str()]:
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
            get_basepairs_for_strands(
                current_motif_strands, self.basepairs, self.chains
            ),
            get_basepair_ends_for_strands(
                current_motif_strands, self.cww_basepairs_lookup.values()
            ),
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
                    for bp in self.cww_residues_to_basepairs[res.get_str()]:
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


if __name__ == "__main__":
    pdb_id = "5UQ7"
    pdb_data = get_pdb_structure_data(pdb_id)
    motif_factory = MotifFactory(pdb_data)
    motifs = motif_factory.get_motifs()
    for m in motifs:
        m.to_cif(f"{m.name}.cif")
    save_motifs_to_json(motifs, f"{pdb_id}_motifs.json")
