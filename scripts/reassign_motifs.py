import os
import random

from typing import List, Dict

from rna_motif_library.motif import (
    get_motifs_from_json,
    are_residues_connected,
    Motif,
    get_chains_from_json,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.residue import Residue
from rna_motif_library.basepair import Basepair, get_basepairs_from_json
from rna_motif_library.util import get_cif_header_str


def sort_rna_residues_by_chain_and_num(residues: List[Residue]) -> List[Residue]:
    """
    Sort residues first by chain ID and then by residue number within each chain.

    Args:
        residues (List[Residue]): List of residues to sort

    Returns:
        List[Residue]: Sorted list of residues
    """
    # Sort using chain ID as primary key and residue number as secondary key
    rna_residues = ["A", "U", "G", "C"]
    keep_residues = []
    for res in residues:
        if res.res_id in rna_residues:
            keep_residues.append(res)
    sorted_residues = sorted(keep_residues, key=lambda x: (x.chain_id, x.num))
    return sorted_residues


class RNAChains:
    def __init__(self, chains: List[List[Residue]]):
        self.chains = chains
        self.residue_dict = {}
        for chain in self.chains:
            for res in chain:
                self.residue_dict[res.get_x3dna_str()] = res

    def get_residues(self) -> List[Residue]:
        residues = []
        for chain in self.chains:
            residues.extend(chain)
        return residues

    def get_residue(self, x3dna_str: str) -> Residue:
        return self.residue_dict[x3dna_str]

    def get_next_residue_in_chain(self, res: Residue) -> Residue:
        chain_num, pos = self.get_residue_position(res)
        if pos < len(self.chains[chain_num]) - 1:
            return self.chains[chain_num][pos + 1]
        return None

    def get_previous_residue_in_chain(self, res: Residue) -> Residue:
        chain_num, pos = self.get_residue_position(res)
        if pos > 0:
            return self.chains[chain_num][pos - 1]
        return None

    def get_residue_by_pos(self, chain_num: int, position: int) -> Residue:
        if chain_num < 0 or chain_num >= len(self.chains):
            return None
        if position < 0 or position >= len(self.chains[chain_num]):
            return None
        return self.chains[chain_num][position]

    def get_residue_position(self, res: Residue) -> int:
        for chain_num, chain in enumerate(self.chains):
            for i, r in enumerate(chain):
                if r == res:
                    return chain_num, i
        return -1, -1

    def are_residues_on_same_chain(self, res1: Residue, res2: Residue) -> bool:
        for chain in self.chains:
            if res1 in chain and res2 in chain:
                return True
        return False

    def get_chain_between_basepair(self, bp: Basepair) -> List[Residue]:
        res1 = self.get_residue(bp.res_1.get_str())
        res2 = self.get_residue(bp.res_2.get_str())
        chain_num_1, pos_1 = self.get_residue_position(res1)
        _, pos_2 = self.get_residue_position(res2)
        if pos_1 > pos_2:
            return self.chains[chain_num_1][pos_2 : pos_1 + 1]
        else:
            return self.chains[chain_num_1][pos_1 : pos_2 + 1]


def get_rna_chains(residues: List[Residue]) -> RNAChains:
    chains = []
    while len(residues) > 0:
        chains.append([residues.pop(0)])
        added = True
        while added:
            added = False
            for chain in chains:
                for res in residues:
                    if are_residues_connected(chain[-1], res) == 1:
                        chain.append(res)
                        residues.remove(res)
                        added = True
                        break
                    if are_residues_connected(chain[0], res) == -1:
                        chain.insert(0, res)
                        residues.remove(res)
                        added = True
                        break
            if not added:
                break
    return RNAChains(chains)


def write_chain_to_cif(chain: List[Residue], filename: str):
    with open(filename, "w") as f:
        f.write(get_cif_header_str())
        acount = 1
        for res in chain:
            s, acount = res.to_cif_str(acount)
            f.write(s)


def get_possible_hairpins(
    chains: RNAChains, basepairs: List[Basepair], allowed_pairs: List[str]
) -> List[List[Residue]]:
    cww_basepairs = {}
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type not in allowed_pairs:
            continue
        cww_basepairs[bp.res_1.get_str() + "-" + bp.res_2.get_str()] = bp
        cww_basepairs[bp.res_2.get_str() + "-" + bp.res_1.get_str()] = bp

    distance = {}
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type not in allowed_pairs:
            continue
        res1 = chains.get_residue(bp.res_1.get_str())
        res2 = chains.get_residue(bp.res_2.get_str())
        if not chains.are_residues_on_same_chain(res1, res2):
            continue
        chain_num_1, pos_1 = chains.get_residue_position(res1)
        chain_num_2, pos_2 = chains.get_residue_position(res2)
        if pos_1 > pos_2:
            pos_1, pos_2 = pos_2, pos_1
        next_res_1 = chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
        next_res_2 = chains.get_residue_by_pos(chain_num_2, pos_2 - 1)
        if (
            next_res_1.get_x3dna_str() + "-" + next_res_2.get_x3dna_str()
            in cww_basepairs
        ):
            continue
        chain = chains.get_chain_between_basepair(bp)
        distance[bp.res_1.get_str() + "-" + bp.res_2.get_str()] = len(chain)

    # Sort basepairs by loop size (distance)
    sorted_pairs = sorted(distance.items(), key=lambda x: x[1])

    hairpins = []
    used_residues = set()

    # Process pairs from smallest to largest loops
    pos = 0
    for pair_str, _ in sorted_pairs:
        # Get the basepair object
        bp = cww_basepairs[pair_str]

        # Get residues in this potential hairpin
        res1 = chains.get_residue(bp.res_1.get_str())
        res2 = chains.get_residue(bp.res_2.get_str())
        chain = chains.get_chain_between_basepair(bp)

        # Skip if any residues already used in smaller hairpins
        overlap = False
        for res in chain:
            if res.get_x3dna_str() in used_residues:
                overlap = True
                break

        if overlap:
            continue

        # Add residues to used set
        for res in chain:
            used_residues.add(res.get_x3dna_str())

        hairpins.append(
            Motif(
                "HAIRPIN-{pos}",
                "HAIRPIN",
                "",
                "",
                "",
                [chain],
                [bp],
                [bp],
                [],
            )
        )
        pos += 1
    return hairpins
    exit()


def get_next_pair_id(bp: Basepair, chains: RNAChains) -> str:
    res1 = chains.get_residue(bp.res_1.get_str())
    res2 = chains.get_residue(bp.res_2.get_str())
    chain_num_1, pos_1 = chains.get_residue_position(res1)
    chain_num_2, pos_2 = chains.get_residue_position(res2)
    next_res1 = chains.get_residue_by_pos(chain_num_1, pos_1 + 1)
    next_res2 = chains.get_residue_by_pos(chain_num_2, pos_2 - 1)
    if next_res1 is None or next_res2 is None:
        return None
    return next_res1.get_x3dna_str() + "-" + next_res2.get_x3dna_str()


def get_prev_pair_id(bp: Basepair, chains: RNAChains) -> str:
    res1 = chains.get_residue(bp.res_1.get_str())
    res2 = chains.get_residue(bp.res_2.get_str())
    chain_num_1, pos_1 = chains.get_residue_position(res1)
    chain_num_2, pos_2 = chains.get_residue_position(res2)
    prev_res1 = chains.get_residue_by_pos(chain_num_1, pos_1 - 1)
    prev_res2 = chains.get_residue_by_pos(chain_num_2, pos_2 + 1)
    if prev_res1 is None or prev_res2 is None:
        return None
    return prev_res1.get_x3dna_str() + "-" + prev_res2.get_x3dna_str()


def get_possible_helices(
    chains: RNAChains,
    basepairs: List[Basepair],
    allowed_pairs: List[str],
    hairpins: List[Motif],
) -> List[List[Residue]]:
    hairpin_residues = []
    for h in hairpins:
        hairpin_residues.extend(h.get_residues())

    # Filter to only cWW basepairs with allowed types
    cww_basepairs = {}
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type not in allowed_pairs:
            continue
        if (
            bp.res_1.get_str() in hairpin_residues
            or bp.res_2.get_str() in hairpin_residues
        ):
            continue
        cww_basepairs[bp.res_1.get_str() + "-" + bp.res_2.get_str()] = bp
        cww_basepairs[bp.res_2.get_str() + "-" + bp.res_1.get_str()] = bp

    helices = []
    used_basepairs = set()

    # Look for consecutive basepairs
    for bp in basepairs:
        if bp.lw != "cWW" or bp.bp_type not in allowed_pairs:
            continue

        prev_pair = get_prev_pair_id(bp, chains)
        # start at beginning of helix
        if prev_pair in cww_basepairs:
            continue

        # Skip if we've already used this basepair in another helix
        if bp.res_1.get_str() + "-" + bp.res_2.get_str() in used_basepairs:
            continue

        # Start a new potential helix
        current_helix = []
        current_bp = bp

        while current_bp is not None:
            current_helix.append(current_bp)
            used_basepairs.add(
                current_bp.res_1.get_str() + "-" + current_bp.res_2.get_str()
            )
            used_basepairs.add(
                current_bp.res_2.get_str() + "-" + current_bp.res_1.get_str()
            )

            # Get residue positions
            next_pair = get_next_pair_id(current_bp, chains)

            if next_pair in cww_basepairs:
                current_bp = cww_basepairs[next_pair]
            else:
                current_bp = None

        helix_residues = []
        if len(current_helix) == 1:
            continue
        strand_1, strand_2 = [], []
        for bp in current_helix:
            res1 = chains.get_residue(bp.res_1.get_str())
            res2 = chains.get_residue(bp.res_2.get_str())
            strand_1.append(res1)
            strand_2.append(res2)
        strand_2 = strand_2[::-1]
        helices.append(
            Motif(
                "HELIX-{pos}",
                "HELIX",
                "",
                "",
                "",
                [strand_1, strand_2],
                current_helix,
                [],
                [],
            )
        )
    return helices


def get_non_helical_strands(
    rna_chains: RNAChains, helices: List[Motif], basepairs: List[Basepair]
) -> List[Motif]:
    # Get all helical residues
    helical_residues = []
    for h in helices:
        helical_residues.extend(h.get_residues())

    # Create dict of cWW basepairs for lookup
    cww_pairs = {}
    for bp in basepairs:
        if bp.lw == "cWW":
            cww_pairs[bp.res_1.get_str()] = bp.res_2.get_str()
            cww_pairs[bp.res_2.get_str()] = bp.res_1.get_str()

    strands = []
    # Process each chain separately to avoid connecting strands across chains
    for chain in rna_chains.chains:
        current_strand = []
        for res in chain:
            if res not in helical_residues:
                # Start or continue a strand
                current_strand.append(res)
            else:
                # End current strand if it exists
                if len(current_strand) > 0:
                    strands.append(current_strand)
                    current_strand = []

        # Add final strand for this chain if it exists
        if len(current_strand) > 0:
            strands.append(current_strand)

    # Check adjacent residues for each strand
    final_strands = []
    for strand in strands:
        # Get first and last residues in strand
        first_res = strand[0]
        last_res = strand[-1]

        # Check previous residue
        chain_num, pos = rna_chains.get_residue_position(first_res)
        if pos > 0:
            prev_res = rna_chains.get_residue_by_pos(chain_num, pos - 1)
            if prev_res.get_x3dna_str() in cww_pairs:
                strand.insert(0, prev_res)

        # Check next residue
        chain_num, pos = rna_chains.get_residue_position(last_res)
        if pos < len(rna_chains.chains[chain_num]) - 1:
            next_res = rna_chains.get_residue_by_pos(chain_num, pos + 1)
            if next_res.get_x3dna_str() in cww_pairs:
                strand.append(next_res)
        final_strands.append(strand)
    return final_strands


def get_missing_residues(
    rna_chains: RNAChains,
    motifs: List[Motif],
) -> List[Residue]:
    motifs_residues = []
    for m in motifs:
        motifs_residues.extend(m.get_residues())
    return [r for r in rna_chains.get_residues() if r not in motifs_residues]


def get_non_canonical_motifs(
    non_helical_strands: List[Motif],
    basepairs: List[Basepair],
    allowed_pairs: List[str],
) -> List[Motif]:
    bp_dict = {}
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type not in allowed_pairs:
            continue
        bp_dict[bp.res_1.get_str()] = bp
        bp_dict[bp.res_2.get_str()] = bp

    # Group strands that share basepairs
    motifs = []
    unprocessed_strands = non_helical_strands.copy()

    count = 0
    while unprocessed_strands:
        # Start a new motif with the first unprocessed strand
        current_strand = unprocessed_strands[0]
        current_motif_strands = [current_strand]
        current_basepair_ends = []
        for res in [current_strand[0], current_strand[-1]]:
            if res.get_x3dna_str() in bp_dict:
                current_basepair_ends.append(bp_dict[res.get_x3dna_str()])

        # Shuffle strands to avoid bias in strand ordering
        random.shuffle(current_motif_strands)
        changed = True
        while changed:
            changed = False

            # Look through remaining strands
            for strand in unprocessed_strands:
                if strand in current_motif_strands:
                    continue
                shares_bp = False
                # Check first and last residue in strand for basepairs with motif
                strand_list = list(strand)
                for res in [strand_list[0], strand_list[-1]]:
                    for end in current_basepair_ends:
                        if (
                            res.get_x3dna_str() == end.res_1.get_str()
                            or res.get_x3dna_str() == end.res_2.get_str()
                        ):
                            shares_bp = True
                            break

                # If strand shares bp, add it to current motif
                if shares_bp:
                    current_motif_strands.append(strand)
                    for res in [strand[0], strand[-1]]:
                        if res.get_x3dna_str() in bp_dict:
                            bp = bp_dict[res.get_x3dna_str()]
                            if bp not in current_basepair_ends:
                                current_basepair_ends.append(bp)
                    changed = True

        for strand in current_motif_strands:
            if strand in unprocessed_strands:
                unprocessed_strands.remove(strand)

        if len(current_motif_strands) > 1:
            end_bp_res = {}
            for bp in current_basepair_ends:
                end_bp_res[bp.res_1.get_str()] = 0
                end_bp_res[bp.res_2.get_str()] = 0

            for strand in current_motif_strands:
                for res in [strand[0], strand[-1]]:
                    if res.get_x3dna_str() in end_bp_res:
                        end_bp_res[res.get_x3dna_str()] = 1

            if len(end_bp_res) != sum(end_bp_res.values()):
                for strand in current_motif_strands:
                    motifs.append(
                        Motif(
                            "UNKNOWN-{pos}",
                            "UNKNOWN",
                            "",
                            "",
                            "",
                            [strand],
                            [],
                            [],
                            [],
                        )
                    )
                continue

        motifs.append(
            Motif(
                "UNKNOWN-{pos}",
                "UNKNOWN",
                "",
                "",
                "",
                current_motif_strands,
                current_basepair_ends,
                current_basepair_ends,
                [],
            )
        )

    return motifs


def get_bulge_or_multiway_junction(
    current_strand: List[Residue],
    strands_between_helices: List[List[Residue]],
    basepairs: List[Basepair],
    allowed_pairs: List[str],
) -> Motif:
    bp_dict = {}
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type not in allowed_pairs:
            continue
        bp_dict[bp.res_1.get_str()] = bp
        bp_dict[bp.res_2.get_str()] = bp

    current_motif_strands = [current_strand]
    current_basepair_ends = []
    for res in [current_strand[0], current_strand[-1]]:
        if res.get_x3dna_str() in bp_dict:
            current_basepair_ends.append(bp_dict[res.get_x3dna_str()])

    changed = True
    while changed:
        changed = False
        # Look through remaining strands
        for strand in strands_between_helices:
            if strand in current_motif_strands:
                continue
            shares_bp = False
            # Check first and last residue in strand for basepairs with motif
            strand_list = list(strand)
            for res in [strand_list[0], strand_list[-1]]:
                for end in current_basepair_ends:
                    if (
                        res.get_x3dna_str() == end.res_1.get_str()
                        or res.get_x3dna_str() == end.res_2.get_str()
                    ):
                        shares_bp = True
                        break

            # If strand shares bp, add it to current motif
            if shares_bp:
                current_motif_strands.append(strand)
                for res in [strand[0], strand[-1]]:
                    if res.get_x3dna_str() in bp_dict:
                        bp = bp_dict[res.get_x3dna_str()]
                        if bp not in current_basepair_ends:
                            current_basepair_ends.append(bp)
                changed = True

    if len(current_motif_strands) > 1:
        end_bp_res = {}
        for bp in current_basepair_ends:
            end_bp_res[bp.res_1.get_str()] = 0
            end_bp_res[bp.res_2.get_str()] = 0

        for strand in current_motif_strands:
            for res in [strand[0], strand[-1]]:
                if res.get_x3dna_str() in end_bp_res:
                    end_bp_res[res.get_x3dna_str()] = 1

        if len(end_bp_res) != sum(end_bp_res.values()):
            return None

    return Motif(
        "UNKNOWN-{pos}",
        "UNKNOWN",
        "",
        "",
        "",
        current_motif_strands,
        current_basepair_ends,
        current_basepair_ends,
        [],
    )


def get_strands_between_helices(
    helices: List[Motif], rna_chains: RNAChains
) -> List[List[Residue]]:
    strands = []
    # Map residues to their helices
    helix_map = {}
    for i, helix in enumerate(helices):
        for res in helix.get_residues():
            helix_map[res.get_x3dna_str()] = i
    # Look for consecutive residues that span different helices
    for chain in rna_chains.chains:
        for i in range(len(chain) - 1):
            res1 = chain[i]
            res2 = chain[i + 1]
            # Skip if either residue isn't in a helix
            if (
                res1.get_x3dna_str() not in helix_map
                or res2.get_x3dna_str() not in helix_map
            ):
                continue
            # Check if residues are in different helices
            if helix_map[res1.get_x3dna_str()] != helix_map[res2.get_x3dna_str()]:
                strands.append([res1, res2])

    return strands


def is_chain_end(motif: Motif, chain_ends: Dict[str, int]) -> bool:
    for res in motif.get_residues():
        if res.get_x3dna_str() in chain_ends:
            return True
    return False


def is_hairpin(motif: Motif, hairpins: List[Motif]) -> bool:
    for h in hairpins:
        count = 0
        h_res = h.get_residues()
        for res in motif.get_residues():
            if res in h_res:
                count += 1
        if count == len(motif.get_residues()):
            return True
    return False


def main():
    allowed_pairs = []
    f = open("end_pairs.txt")
    lines = f.readlines()
    for line in lines:
        allowed_pairs.append(line.strip())
    f.close()
    pdb_code = "4P95"
    json_path = os.path.join(DATA_PATH, "jsons", "chains", f"{pdb_code}.json")
    chains = get_chains_from_json(json_path)
    rna_chains = RNAChains(chains)
    json_path = os.path.join(DATA_PATH, "jsons", "basepairs", f"{pdb_code}.json")
    basepairs = get_basepairs_from_json(json_path)
    hairpins = get_possible_hairpins(rna_chains, basepairs, allowed_pairs)
    helices = get_possible_helices(rna_chains, basepairs, allowed_pairs, hairpins)
    strands_between_helices = get_strands_between_helices(helices, rna_chains)
    for i, s in enumerate(strands_between_helices):
        write_chain_to_cif(s, f"strand_between_helices_{i}.cif")
    non_helical_strands = get_non_helical_strands(rna_chains, helices, basepairs)
    non_canonical_motifs = get_non_canonical_motifs(
        non_helical_strands, basepairs, allowed_pairs
    )
    missing_residues = get_missing_residues(rna_chains, helices + non_canonical_motifs)
    if len(missing_residues) > 0:
        print("Missing residues:", missing_residues)
        exit()

    chain_ends = {}
    for chain in rna_chains.chains:
        chain_ends[chain[0].get_x3dna_str()] = 1
        chain_ends[chain[-1].get_x3dna_str()] = 1
    finished_motifs = []
    finished_motifs.extend(hairpins)
    finished_motifs.extend(helices)
    unfinished_motifs = []
    count = -1
    for m in non_canonical_motifs:
        count += 1
        if is_chain_end(m, chain_ends):
            m.mtype = "SSTRAND"
            m.name = f"SSTRAND-{count}"
            finished_motifs.append(m)
            continue
        if is_hairpin(m, hairpins):
            m.mtype = "HAIRPIN"
            m.name = f"HAIRPIN-{count}"
            finished_motifs.append(m)
            continue
        if len(m.strands) > 1:
            m.mtype = "SSTRAND"
            m.name = f"SSTRAND-{count}"
            finished_motifs.append(m)
            continue
        unfinished_motifs.append(m)

    pos = 0
    for m in unfinished_motifs:
        new_motif = get_bulge_or_multiway_junction(
            m.strands[0], strands_between_helices, basepairs, allowed_pairs
        )
        if new_motif is None:
            finished_motifs.append(m)
            continue
        if len(new_motif.get_residues()) > len(m.get_residues()):
            finished_motifs.append(new_motif)
            pos += 1
        else:
            finished_motifs.append(m)
    for i, m in enumerate(finished_motifs):
        m.to_cif(f"motif_{i}.cif")
    missing_residues = get_missing_residues(rna_chains, finished_motifs)
    for r in missing_residues:
        print(r.get_x3dna_str())


if __name__ == "__main__":
    main()
