import pandas as pd
from collections import defaultdict

from rna_motif_library.residue import get_cached_residues
from rna_motif_library.util import parse_residue_identifier


def main():
    residues = get_cached_residues("1GID")
    num_atoms = defaultdict(list)
    for k, v in residues.items():
        num_atoms[v.res_id].append(len(v.coords))

    for k, v in num_atoms.items():
        print(k, sum(v) / len(v))


if __name__ == "__main__":
    main()
