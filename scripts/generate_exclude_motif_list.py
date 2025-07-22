import pandas as pd
import os

def main():
    exclude_pdb_ids = [
        "7ASO",
        "7ASP",
        "6R7G",
    ]
    path = os.path.join(
        "scripts",
        "resources",
        "exclude_list",
    )
    df = pd.read_json(
        os.path.join(path, "unique_tertiary_contacts.json")
    )
    df = df[~df["pdb_id"].isin(exclude_pdb_ids)]
    df.sort_values("num_hbonds", ascending=False, inplace=True)
    df_exclude = pd.read_csv(os.path.join(path, "large_motifs_gt_50.csv"))
    df_exclude = df_exclude.query("exclude == 1")
    exclude_motif_ids = list(df_exclude["motif_id"].values)
    # manual removals from manual inspection
    tc_removals = [1, 2, 10, 14, 19, 21, 22, 24, 31, 34, 36, 37, 38, 39, 4, 40, 42, 47, 49, 52, 57, 59, 6, 61, 63, 7, 76, 8, 81, 88, 96, 99]
    for i in tc_removals:
        row = df.iloc[i]
        exclude_motif_ids.append(row["motif_1_id"])
        exclude_motif_ids.append(row["motif_2_id"])
        print(row["motif_1_id"], row["motif_2_id"])
    df_exclude = pd.DataFrame({"motif_id": exclude_motif_ids})
    df_exclude.to_csv("data/summaries/exclude_motif_list.csv", index=False)

if __name__ == "__main__":
    main()