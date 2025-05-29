import os
from rna_motif_library.motif import get_motifs_from_json, get_cached_motifs

from rna_motif_library.motif_factory import (
    HelixFinder,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    get_cww_basepairs,
)


def compare_motif_sets(motifs_1, motifs_2):
    only_in_1 = []
    only_in_2 = []
    seen_in_1 = set()
    seen_in_2 = set()

    # Find motifs only in motifs_1
    for m1 in motifs_1:
        found = False
        m1_residues = set(res.get_str() for res in m1.get_residues())

        for m2 in motifs_2:
            m2_residues = set(res.get_str() for res in m2.get_residues())
            if m1_residues == m2_residues:
                seen_in_1.add(m1.name)
                seen_in_2.add(m2.name)
                found = True
                break
        if not found:
            only_in_1.append(m1)

    # Find motifs only in motifs_2
    for m2 in motifs_2:
        if m2.name not in seen_in_2:
            only_in_2.append(m2)

    return only_in_1, only_in_2


def find_helices_in_motifs(motifs):
    pdb_data = get_pdb_structure_data("4V8Q")
    cww_basepairs = get_cww_basepairs(
        pdb_data, min_two_hbond_score=0.5, min_three_hbond_score=0.5
    )
    count = 0
    for m in motifs:
        if m.mtype == "HELIX":
            continue
        res = m.get_residues()
        pdb_data_for_residues = get_pdb_structure_data_for_residues(pdb_data, res)
        hf = HelixFinder(pdb_data_for_residues, cww_basepairs, [])
        helices = hf.get_helices()
        if len(helices) > 0:
            print(m.name, len(helices), count)
            count += 1


def main():
    pdb_id = "4WZD"
    motifs_1 = get_cached_motifs(pdb_id)
    motifs_2 = get_motifs_from_json(f"4WZD_bak.json")
    find_helices_in_motifs(motifs_2)
    only_in_1, only_in_2 = compare_motif_sets(motifs_1, motifs_2)
    print(len(only_in_1))
    print(len(only_in_2))
    os.makedirs("only_in_1", exist_ok=True)
    os.makedirs("only_in_2", exist_ok=True)
    for m in only_in_1:
        m.to_cif(f"only_in_1/{m.name}-only-1.cif")
    for m in only_in_2:
        m.to_cif(f"only_in_2/{m.name}-only-2.cif")


if __name__ == "__main__":
    main()
