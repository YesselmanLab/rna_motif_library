import pandas as pd
import glob
import pickle

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.tranforms import align_motif


def check_for_no_matches():
    csvs = glob.glob("data/dataframes/duplicate_motifs/*.csv")
    num_no_matches = 0
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except:
            continue
        df_sum = df.query("rmsd == 1000")
        df_sum = df_sum.query("not motif.str.startswith('SSTRAND-1')")
        if len(df_sum) > 0:
            print(csv, len(df_sum))
            num_no_matches += len(df_sum)
    print(num_no_matches)


def process_hairpins():
    df = pd.read_csv("data/summaries/non_redundant_motifs.csv")
    unique_motifs = df["motif"].values
    hairpins = pickle.load(open("twoways.pkl", "rb"))
    sequence = "CAG-CG"
    ref_motif = None
    target_motifs = []
    for pdb_id, motifs in hairpins.items():
        for motif in motifs:
            if motif.name == "TWOWAY-1-0-CAG-CG-7SFR-4":
                ref_motif = motif
            if motif.name not in unique_motifs:
                continue
            if motif.sequence == sequence:
                target_motifs.append(motif)
    ref_motif.to_cif("ref_m.cif")
    count = 0
    for m in target_motifs:
        aligned_m = align_motif(m, ref_motif)
        if aligned_m is None:
            continue
        aligned_m.to_cif(f"aligned_{count}.cif")
        count += 1


def get_rmsd_histogram():
    csv_files = glob.glob("data/dataframes/non_redundant_sets/*.csv")
    ref_motif = "HAIRPIN-4-CUACGG-6SWD-1"
    dfs = []
    for csv in csv_files:
        try:
            df = pd.read_csv(csv)
        except:
            continue
        df = df.query("is_duplicate == False")
        df = df.query("motif.str.contains('HAIRPIN-4-CUACGG')")
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("hairpin_rmsd.csv", index=False)
    print(len(df))


def main():
    process_hairpins()
    exit()
    df = pd.read_csv("data/dataframes/duplicate_motifs/NR_3.5_98632.6.csv")
    df = df.query("is_duplicate == False")
    all_motifs = {}
    pdb_ids = list(df["repr_pdb"].unique()) + list(df["child_pdb"].unique())
    for pdb_id in pdb_ids:
        motifs = get_cached_motifs(pdb_id)
        for motif in motifs:
            all_motifs[motif.name] = motif
    for i, g in df.groupby("repr_motif"):
        if len(g) < 5:
            continue
        print(i, len(g))
        ref_m = all_motifs[i]
        ref_m.to_cif("ref_m.cif")
        count = 0
        for j, row in g.iterrows():
            child_m = all_motifs[row["motif"]]
            aligned_m = align_motif(child_m, ref_m)
            if aligned_m is None:
                continue
            aligned_m.to_cif(f"aligned_{count}.cif")
            count += 1
        exit()


if __name__ == "__main__":
    main()
