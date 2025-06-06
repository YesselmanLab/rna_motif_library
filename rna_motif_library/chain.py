import json
import os
import numpy as np
from typing import Set, List, Dict, Tuple, Optional

from rna_motif_library.residue import (
    Residue,
    are_residues_connected,
    are_protein_residues_connected,
)
from rna_motif_library.basepair import Basepair
from rna_motif_library.util import (
    get_cif_header_str,
    wc_basepairs_w_gu,
    get_cached_path,
    canon_amino_acid_list,
)
from rna_motif_library.logger import get_logger

log = get_logger("chain")


class Chains:
    def __init__(self, chains: List[List[Residue]], ctype: str = "RNA"):
        """Initialize Chains with a list of residue chains.

        Args:
            chains: List of lists of Residue objects representing chains of residues
        """
        self.chains = chains
        self.ctype = ctype
        # Create lookup dict for O(1) residue access by res_str
        self.residue_dict = {res.get_str(): res for chain in chains for res in chain}

        # Create lookup dict for O(1) residue position access
        self.position_dict = {}
        for chain_num, chain in enumerate(chains):
            for pos, res in enumerate(chain):
                self.position_dict[res] = (chain_num, pos)
        self.chain_ends = {}
        for chain in self.chains:
            self.chain_ends[chain[0].get_str()] = 1
            self.chain_ends[chain[-1].get_str()] = 1

    def get_residues(self) -> List[Residue]:
        """Get all residues across all chains.

        Returns:
            List of all Residue objects
        """
        return list(self.residue_dict.values())

    def get_residue_by_str(self, res_str: str) -> Residue:
        """Get residue by string identifier.

        Args:
            res_str: string identifier

        Returns:
            Matching Residue object
        """
        if res_str not in self.residue_dict:
            return None
        else:
            return self.residue_dict[res_str]

    def get_chain_for_residue(self, res: Residue) -> Optional[List[Residue]]:
        try:
            chain_num, pos = self.position_dict[res]
            return self.chains[chain_num]
        except KeyError:
            return None

    def get_next_residue_in_chain(self, res: Residue) -> Optional[Residue]:
        """Get next residue in chain after given residue.

        Args:
            res: Current residue

        Returns:
            Next residue in chain or None if at end
        """
        chain_num, pos = self.position_dict[res]
        if pos < len(self.chains[chain_num]) - 1:
            return self.chains[chain_num][pos + 1]
        return None

    def get_previous_residue_in_chain(self, res: Residue) -> Optional[Residue]:
        """Get previous residue in chain before given residue.

        Args:
            res: Current residue

        Returns:
            Previous residue in chain or None if at start
        """
        chain_num, pos = self.position_dict[res]
        if pos > 0:
            return self.chains[chain_num][pos - 1]
        return None

    def get_residue_by_pos(self, chain_num: int, position: int) -> Optional[Residue]:
        """Get residue at specific chain and position.

        Args:
            chain_num: Chain index
            position: Position in chain

        Returns:
            Residue at position or None if invalid position
        """
        if 0 <= chain_num < len(self.chains):
            chain = self.chains[chain_num]
            if 0 <= position < len(chain):
                return chain[position]
        return None

    def get_residue_position(self, res: Residue) -> Tuple[int, int]:
        """Get chain number and position of residue.

        Args:
            res: Residue to locate

        Returns:
            Tuple of (chain_num, position) or (-1,-1) if not found
        """
        return self.position_dict.get(res, (-1, -1))

    def are_residues_on_same_chain(self, res1: Residue, res2: Residue) -> bool:
        """Check if two residues are on the same chain.

        Args:
            res1: First residue
            res2: Second residue

        Returns:
            True if residues are on same chain
        """
        chain_num1 = self.position_dict[res1][0]
        chain_num2 = self.position_dict[res2][0]
        return chain_num1 == chain_num2

    def get_chain_between_basepair(self, bp: Basepair) -> List[Residue]:
        """Get residues between two basepaired residues on same chain.

        Args:
            bp: Basepair object

        Returns:
            List of residues between basepair
        """
        res1 = self.get_residue_by_str(bp.res_1.get_str())
        res2 = self.get_residue_by_str(bp.res_2.get_str())
        chain_num_1, pos_1 = self.position_dict[res1]
        _, pos_2 = self.position_dict[res2]
        start, end = min(pos_1, pos_2), max(pos_1, pos_2)
        return self.chains[chain_num_1][start : end + 1]

    def get_residues_in_basepair(self, bp: Basepair) -> List[Residue]:
        """Get residues in a basepair."""
        try:
            res1 = self.get_residue_by_str(bp.res_1.get_str())
            res2 = self.get_residue_by_str(bp.res_2.get_str())
            return [res1, res2]
        except Exception as e:
            return []

    def is_chain_end(self, res: Residue) -> bool:
        return res.get_str() in self.chain_ends

    def get_prev_pair_id(self, bp: Basepair) -> Optional[str]:
        res_1, res_2 = self.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.get_residue_position(res_1)
        chain_num_2, pos_2 = self.get_residue_position(res_2)
        prev_res1 = self.get_residue_by_pos(chain_num_1, pos_1 - 1)
        prev_res2 = self.get_residue_by_pos(chain_num_2, pos_2 + 1)
        if prev_res1 is None or prev_res2 is None:
            return None
        return prev_res1.get_str() + "-" + prev_res2.get_str()

    def get_next_pair_id(self, bp: Basepair) -> Optional[str]:
        res_1, res_2 = self.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.get_residue_position(res_1)
        chain_num_2, pos_2 = self.get_residue_position(res_2)
        next_res_1 = self.get_residue_by_pos(chain_num_1, pos_1 + 1)
        next_res_2 = self.get_residue_by_pos(chain_num_2, pos_2 - 1)
        if next_res_1 is None or next_res_2 is None:
            return None
        return next_res_1.get_str() + "-" + next_res_2.get_str()

    def get_possible_flanking_bps(self, bp: Basepair) -> List[Basepair]:
        res1, res2 = self.get_residues_in_basepair(bp)
        chain_num_1, pos_1 = self.get_residue_position(res1)
        chain_num_2, pos_2 = self.get_residue_position(res2)
        res1_next = self.get_residue_by_pos(chain_num_1, pos_1 + 1)
        res1_prev = self.get_residue_by_pos(chain_num_1, pos_1 - 1)
        res2_next = self.get_residue_by_pos(chain_num_2, pos_2 + 1)
        res2_prev = self.get_residue_by_pos(chain_num_2, pos_2 - 1)
        combos = [
            (res1_next, res2_prev),
            (res1_next, res2_next),
            (res1_prev, res2_prev),
            (res1_prev, res2_next),
        ]
        return combos


def p5_to_p3_connection_distance(res1: Residue, res2: Residue) -> Optional[float]:
    o3_coords_1 = res1.get_atom_coords("O3'")
    p_coords_2 = res2.get_atom_coords("P")
    if p_coords_2 is None:
        p_coords_2 = res2.get_atom_coords("PA")
    if o3_coords_1 is not None and p_coords_2 is not None:
        distance = np.linalg.norm(np.array(p_coords_2) - np.array(o3_coords_1))
        return distance
    return None


def find_chain_ends(res_list: List[Residue]) -> List[Residue]:
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
    return roots


def sort_rna_residues_by_chain_and_num(residues: List[Residue]) -> List[Residue]:
    """
    Sort residues first by chain ID and then by residue number within each chain.

    Args:
        residues (List[Residue]): List of residues to sort

    Returns:
        List[Residue]: Sorted list of residues
    """
    # Sort using chain ID as primary key and residue number as secondary key
    removed_ids = []
    rna_residues = ["A", "U", "G", "C"]
    keep_residues = []
    for res in residues:
        if res.res_id in rna_residues:
            keep_residues.append(res)
        elif res.get_atom_coords("P") is not None:
            keep_residues.append(res)
        elif res.get_atom_coords("C1'") is not None:
            keep_residues.append(res)
        else:
            if res.res_id not in removed_ids:
                removed_ids.append(res.res_id)
    sorted_residues = sorted(keep_residues, key=lambda x: (x.chain_id, x.num))
    return sorted_residues


def sort_protein_residues_by_chain_and_num(residues: List[Residue]) -> List[Residue]:
    """
    Sort protein residues first by chain ID and then by residue number within each chain.

    Args:
        residues (List[Residue]): List of residues to sort

    Returns:
        List[Residue]: Sorted list of residues
    """
    # Sort using chain ID as primary key and residue number as secondary key
    keep_residues = []
    for res in residues:
        if res.res_id in canon_amino_acid_list:
            keep_residues.append(res)
            continue
        # Check for protein backbone atoms
        if (
            res.get_atom_coords("C") is not None
            and res.get_atom_coords("N") is not None
        ):
            keep_residues.append(res)
    sorted_residues = sorted(keep_residues, key=lambda x: (x.chain_id, x.num))
    return sorted_residues


def sort_rna_residues_by_chain_and_num(residues: List[Residue]) -> List[Residue]:
    """
    Sort residues first by chain ID and then by residue number within each chain.

    Args:
        residues (List[Residue]): List of residues to sort

    Returns:
        List[Residue]: Sorted list of residues
    """
    # Sort using chain ID as primary key and residue number as secondary key
    removed_ids = []
    rna_residues = ["A", "U", "G", "C"]
    keep_residues = []
    for res in residues:
        if res.res_id in rna_residues:
            keep_residues.append(res)
        elif res.get_atom_coords("P") is not None:
            keep_residues.append(res)
        elif res.get_atom_coords("C1'") is not None:
            keep_residues.append(res)
        else:
            if res.res_id not in removed_ids:
                removed_ids.append(res.res_id)
    sorted_residues = sorted(keep_residues, key=lambda x: (x.chain_id, x.num))
    return sorted_residues


def get_diff_in_residue_num(res1: Residue, res2: Residue) -> int:
    return res1.num - res2.num


def get_rna_chains(residues: List[Residue]) -> List[List[Residue]]:
    """Group RNA residues into connected chains.

    Takes a list of residues and groups them into chains based on connectivity.
    Residues are considered connected if they share a phosphate backbone.

    Args:
        residues: List of RNA residues to group into chains

    Returns:
        RNAChains object containing the grouped chains
    """
    residues = sort_rna_residues_by_chain_and_num(residues)
    chains = []
    while residues:
        # Start a new chain with first remaining residue
        current_chain = [residues.pop(0)]
        chains.append(current_chain)
        # Keep extending chain until no more connections found
        while True:
            found_connection = False
            # Try to extend chain in both directions
            for res in residues:
                if res.chain_id != current_chain[0].chain_id:
                    continue
                if are_residues_connected(current_chain[-1], res) == 1:
                    current_chain.append(res)
                    residues.remove(res)
                    found_connection = True
                    break
                if are_residues_connected(current_chain[0], res) == -1:
                    current_chain.insert(0, res)
                    residues.remove(res)
                    found_connection = True
                    break
            if not found_connection:
                break
    # Keep merging chains until no more merges are possible
    while True:
        merged_chains = []
        i = 0
        merged_any = False

        while i < len(chains):
            # Handle last chain
            if i == len(chains) - 1:
                merged_chains.append(chains[i])
                break

            chain_1 = chains[i]
            chain_2 = chains[i + 1]
            res_1 = chain_1[-1]
            res_2 = chain_2[0]

            merge = False
            if res_1.chain_id == res_2.chain_id:
                if get_diff_in_residue_num(res_2, res_1) == 1:
                    distance = p5_to_p3_connection_distance(res_1, res_2)
                    if distance is None:
                        res_1_mean = np.mean(res_1.get_sugar_atom_coords(), axis=0)
                        res_2_mean = np.mean(res_2.get_sugar_atom_coords(), axis=0)
                        distance = np.linalg.norm(res_2_mean - res_1_mean)
                        if distance < 8:
                            log.info(
                                f"Merging chains with {res_1.get_x3dna_str()} and {res_2.get_x3dna_str()}"
                            )
                            merge = True
                    elif distance < 5:
                        log.info(
                            f"Merging chains with {res_1.get_x3dna_str()} and {res_2.get_x3dna_str()}"
                        )
                        merge = True

            if merge:
                merged_chains.append(chain_1 + chain_2)
                merged_any = True
                i += 2
            else:
                merged_chains.append(chain_1)
                i += 1

        chains = merged_chains

        if not merged_any:
            break

    return chains


def get_protein_chains(residues: List[Residue]) -> List[List[Residue]]:
    """Group protein residues into connected chains.

    Takes a list of residues and groups them into chains based on connectivity.
    Residues are considered connected if they share a peptide bond.

    Args:
        residues: List of RNA residues to group into chains

    Returns:
        RNAChains object containing the grouped chains
    """
    residues = sort_protein_residues_by_chain_and_num(residues)
    chains = []
    while residues:
        # Start a new chain with first remaining residue
        current_chain = [residues.pop(0)]
        chains.append(current_chain)
        # Keep extending chain until no more connections found
        while True:
            found_connection = False
            # Try to extend chain in both directions
            for res in residues:
                if res.chain_id != current_chain[0].chain_id:
                    continue
                if are_protein_residues_connected(current_chain[-1], res) == 1:
                    current_chain.append(res)
                    residues.remove(res)
                    found_connection = True
                    break
                if are_protein_residues_connected(current_chain[0], res) == -1:
                    current_chain.insert(0, res)
                    residues.remove(res)
                    found_connection = True
                    break
            if not found_connection:
                break
    return chains


def save_chains_to_json(chains: List[List[Residue]], output_path: str) -> None:
    """Save RNA chains to JSON file.

    Args:
        chains: List of lists of Residue objects representing RNA chains
        output_path: Path to save JSON file
    """
    # Convert chains to serializable format
    json_chains = []
    for chain in chains:
        json_chain = []
        for res in chain:
            json_chain.append(res.to_dict())
        json_chains.append(json_chain)

    with open(output_path, "w") as f:
        json.dump(json_chains, f, indent=2)


def do_strands_have_helix_sequence(strands: List[List[Residue]]) -> bool:
    strand_1 = strands[0]
    strand_2 = strands[1]
    for res1, res2 in zip(strand_1, strand_2[::-1]):
        if res1.res_id + res2.res_id not in wc_basepairs_w_gu:
            return False
    return True


def get_chains_from_json(json_path: str) -> List[List[Residue]]:
    """Load RNA chains from JSON file.

    Args:
        json_path: Path to JSON file containing chain data

    Returns:
        List of lists of Residue objects representing RNA chains
    """
    with open(json_path) as f:
        json_chains = json.load(f)

    chains = []
    for json_chain in json_chains:
        chain = []
        for res_dict in json_chain:
            res = Residue.from_dict(res_dict)
            chain.append(res)
        chains.append(chain)

    return chains


def write_chain_to_cif(chain: List[Residue], filename: str):
    with open(filename, "w") as f:
        f.write(get_cif_header_str())
        acount = 1
        for res in chain:
            s = res.to_cif_str(acount)
            acount += len(res.atom_names)
            f.write(s)


def get_cached_chains(pdb_id: str) -> List[List[Residue]]:
    json_path = get_cached_path(pdb_id, "chains")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Chains file not found for {pdb_id}")
    return get_chains_from_json(json_path)


def get_cached_protein_chains(pdb_id: str) -> List[List[Residue]]:
    json_path = get_cached_path(pdb_id, "protein_chains")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Protein chains file not found for {pdb_id}")
    return get_chains_from_json(json_path)
