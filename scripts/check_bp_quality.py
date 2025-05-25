import pandas as pd
from rna_motif_library.basepair import get_cached_basepairs


def main():
    basepairs = get_cached_basepairs("7UO0")
    avg_score = 0
    count = 0
    for bp in basepairs:
        if bp.lw != "cWW":
            continue
        if bp.bp_type == "C-G" or bp.bp_type == "G-C":
            avg_score += bp.hbond_score / 3
        elif bp.bp_type == "A-U" or bp.bp_type == "U-A":
            avg_score += bp.hbond_score / 2
        else:
            continue
        count += 1
    print(avg_score / count)


if __name__ == "__main__":
    main()
