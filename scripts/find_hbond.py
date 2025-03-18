import pandas as pd
from rna_motif_library.residue import get_cached_residues
from rna_motif_library.hbond import HbondFactory, score_hbond


def main():
    pdb_code = "4P95"
    residues = get_cached_residues(pdb_code)
    res_1 = residues["A-A-275-"]
    res_2 = residues["A-C-345-"]
    hf = HbondFactory()
    hbonds = hf.find_hbonds(res_1, res_2, pdb_code)
    print("finished")
    for hbond in hbonds:
        print(
            hbond.res_1.get_str(),
            hbond.res_2.get_str(),
            hbond.atom_1,
            hbond.atom_2,
            hbond.distance,
            hbond.angle_1,
            hbond.angle_2,
            hbond.dihedral_angle,
        )
        print(
            score_hbond(
                hbond.distance,
                hbond.angle_1,
                hbond.angle_2,
                hbond.dihedral_angle,
            )
        )


if __name__ == "__main__":
    main()
