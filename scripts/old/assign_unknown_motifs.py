import pandas as pd
import os
from typing import List, Dict
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.classes import (
    get_residues_from_json,
    get_basepairs_from_json,
    Residue,
)
from rna_motif_library.motif import (
    get_motifs_from_json,
    are_residues_connected,
    find_chain_ends,
    Motif,
    MotifFactory,
)
from rna_motif_library.util import wc_basepairs_w_gu


def do_strands_have_helix_sequence(strands: List[List[Residue]]) -> bool:
    strand_1 = strands[0]
    strand_2 = strands[1]
    for res1, res2 in zip(strand_1, strand_2[::-1]):
        if res1.res_id + res2.res_id not in wc_basepairs_w_gu:
            return False
    return True


def are_end_residues_chain_ends(
    residues: Dict[str, Residue], strands: List[List[Residue]]
) -> bool:
    strand_ends = []
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


def split_motif_into_single_strands(motif: Motif, mf: MotifFactory) -> List[Motif]:
    strands = motif.strands
    new_motifs = []
    for strand in strands:
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
        if len(res) == 0:
            continue
        new_motif = mf.from_residues(res)
        new_motifs.append(new_motif)
    return new_motifs


def find_residues_outside_motifs(residues, motifs):
    residues_in_motifs = {}
    for motif in motifs:
        for res in motif.get_residues():
            residues_in_motifs[res.get_x3dna_str()] = res
    count = 0
    for res in residues.values():
        if res.get_x3dna_str() in residues_in_motifs:
            continue
        if res.res_id in ["A", "G", "C", "U"]:
            count += 1
            print(res.get_x3dna_str())
    print(count)


def main():
    pdb_code = "6ZJ3"
    hbonds = []
    json_path = os.path.join(DATA_PATH, "jsons", "basepairs", f"{pdb_code}.json")
    basepairs = get_basepairs_from_json(json_path)
    mf = MotifFactory(pdb_code, hbonds, basepairs)
    json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
    motifs = get_motifs_from_json(json_path)
    json_path = os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
    residues = get_residues_from_json(json_path)
    for motif in motifs:
        if motif.mtype != "UNKNOWN":
            continue
        if motif.name != "6ZJ3-UNKNOWN-XG-CC-1":
            continue
        print(motif.name)
        motif.to_cif(os.path.join(f"unknowns/{motif.name}.cif"))


if __name__ == "__main__":
    main()
