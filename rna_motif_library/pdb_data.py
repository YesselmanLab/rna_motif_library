from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Set

import click

from rna_motif_library.basepair import Basepair
from rna_motif_library.chain import Chains, get_cached_chains, get_rna_chains
from rna_motif_library.hbond import Hbond, get_cached_hbonds
from rna_motif_library.logger import get_logger
from rna_motif_library.residue import Residue, get_cached_residues
from rna_motif_library.settings import RESOURCES_PATH
from rna_motif_library.basepair import get_cached_basepairs

log = get_logger("pdb_data")


@dataclass
class PDBStructureData:
    pdb_id: str
    chains: Chains
    residues: List[Residue]
    basepairs: List[Basepair]
    hbonds: List[Hbond]


def get_pdb_structure_data(pdb_id: str) -> PDBStructureData:
    chains = Chains(get_cached_chains(pdb_id))
    basepairs = get_cached_basepairs(pdb_id)
    residues = get_cached_residues(pdb_id)
    hbonds = get_cached_hbonds(pdb_id)
    return PDBStructureData(pdb_id, chains, residues, basepairs, hbonds)


def get_pdb_structure_data_for_residues(
    pdb_data: PDBStructureData, residues: List[Residue]
) -> PDBStructureData:
    chains = Chains(get_rna_chains(residues))
    basepairs = get_basepairs_for_residue(residues, pdb_data.basepairs)
    residue_dict = {r.get_str(): r for r in residues}
    return PDBStructureData(
        pdb_data.pdb_id, chains, residue_dict, basepairs, pdb_data.hbonds
    )


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
    min_two_hbond_score: float = 0.5,
    min_three_hbond_score: float = 0.5,
) -> Dict[str, Basepair]:
    """Get dictionary of cWW (cis Watson-Watson) basepairs from a PDB structure.

    Args:
        pdb_data: PDBStructureData object containing basepairs and chains information
        valid_pairs: Optional list of valid basepair types (e.g. ['A-U', 'G-C']). If None, loads from file.
        min_two_hbond_score: Minimum required hbond score for A-U and G-U pairs (default 0.5)
        min_three_hbond_score: Minimum required hbond score for G-C pairs (default 0.5)

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
        # make sure its in chains, i.e. part of the structure
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


def get_singlet_pairs(
    cww_basepairs: Dict[str, Basepair], chains: Chains
) -> Dict[str, Basepair]:
    singlet_pairs = {}
    for bp in cww_basepairs.values():
        combos = chains.get_possible_flanking_bps(bp)
        count = 0
        for res1, res2 in combos:
            if res1 is None or res2 is None:
                continue
            key1 = f"{res1.get_str()}-{res2.get_str()}"
            key2 = f"{res2.get_str()}-{res1.get_str()}"
            if key1 in cww_basepairs or key2 in cww_basepairs:
                count += 1
        if count == 0:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            singlet_pairs[key] = bp
    return singlet_pairs


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pdb_id", type=str)
@click.option("--min_hbond_score", type=float, default=0.0)
@click.option("-r1", "--res1", type=str, default=None)
@click.option("-r2", "--res2", type=str, default=None)
def list_basepairs(pdb_id, min_hbond_score, res1, res2):
    pdb_data = get_pdb_structure_data(pdb_id)
    for bp in pdb_data.basepairs:
        if bp.hbond_score < min_hbond_score:
            continue
        if (
            res1 is not None
            and bp.res_1.get_str() != res1
            and bp.res_2.get_str() != res1
        ):
            continue
        if (
            res2 is not None
            and bp.res_2.get_str() != res2
            and bp.res_1.get_str() != res2
        ):
            continue
        print(bp.res_1.get_str(), bp.res_2.get_str(), bp.bp_type, bp.hbond_score, bp.lw)


if __name__ == "__main__":
    cli()
