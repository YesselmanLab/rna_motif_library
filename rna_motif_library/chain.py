import json
import numpy as np
from typing import Set, List, Dict, Tuple, Optional

from rna_motif_library.residue import Residue
from rna_motif_library.basepair import Basepair
from rna_motif_library.util import get_cif_header_str, wc_basepairs_w_gu


class RNAChains:
    def __init__(self, chains: List[List[Residue]]):
        """Initialize RNAChains with a list of residue chains.

        Args:
            chains: List of lists of Residue objects representing RNA chains
        """
        self.chains = chains
        # Create lookup dict for O(1) residue access by x3dna string
        self.residue_dict = {
            res.get_x3dna_str(): res for chain in chains for res in chain
        }

        # Create lookup dict for O(1) residue position access
        self.position_dict = {}
        for chain_num, chain in enumerate(chains):
            for pos, res in enumerate(chain):
                self.position_dict[res] = (chain_num, pos)
        self.chain_ends = {}
        for chain in self.chains:
            self.chain_ends[chain[0].get_x3dna_str()] = 1
            self.chain_ends[chain[-1].get_x3dna_str()] = 1

    def get_residues(self) -> List[Residue]:
        """Get all residues across all chains.

        Returns:
            List of all Residue objects
        """
        return list(self.residue_dict.values())

    def get_residue(self, x3dna_str: str) -> Residue:
        """Get residue by x3dna string identifier.

        Args:
            x3dna_str: x3dna format string identifier

        Returns:
            Matching Residue object
        """
        if x3dna_str not in self.residue_dict:
            return None
        else:
            return self.residue_dict[x3dna_str]

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
        res1 = self.get_residue(bp.res_1.get_str())
        res2 = self.get_residue(bp.res_2.get_str())
        chain_num_1, pos_1 = self.position_dict[res1]
        _, pos_2 = self.position_dict[res2]
        start, end = min(pos_1, pos_2), max(pos_1, pos_2)
        return self.chains[chain_num_1][start : end + 1]

    def get_residues_in_basepair(self, bp: Basepair) -> List[Residue]:
        """Get residues in a basepair."""
        try:
            res1 = self.get_residue(bp.res_1.get_str())
            res2 = self.get_residue(bp.res_2.get_str())
            return [res1, res2]
        except Exception as e:
            return []

    def is_chain_end(self, res: Residue) -> bool:
        return res.get_x3dna_str() in self.chain_ends


def are_residues_connected(
    source_residue: Residue,
    residue_in_question: Residue,
    cutoff: float = 2.75,
) -> int:
    """Determine if another residue is connected to this residue"""
    # Get O3' coordinates from source residue
    o3_coords_1 = source_residue.get_atom_coords("O3'")
    p_coords_2 = residue_in_question.get_atom_coords("P")
    # check for Triphosphates
    if p_coords_2 is None:
        p_coords_2 = residue_in_question.get_atom_coords("PA")

    # Check 5' to 3' connection
    if o3_coords_1 is not None and p_coords_2 is not None:
        distance = np.linalg.norm(np.array(p_coords_2) - np.array(o3_coords_1))
        if distance < cutoff:
            return 1

    # Check 3' to 5' connection
    o3_coords_2 = residue_in_question.get_atom_coords("O3'")
    p_coords_1 = source_residue.get_atom_coords("P")
    if p_coords_1 is None:
        p_coords_1 = source_residue.get_atom_coords("PA")

    if o3_coords_2 is not None and p_coords_1 is not None:
        distance = np.linalg.norm(np.array(o3_coords_2) - np.array(p_coords_1))
        if distance < cutoff:
            return -1

    return 0


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
            s, acount = res.to_cif_str(acount)
            f.write(s)
