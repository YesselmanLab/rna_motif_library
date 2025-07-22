import pandas as pd


def main():
    df = pd.read_csv("9MDS_old_check_motifs.csv")
    df_new = pd.read_csv("data/dataframes/check_motifs/9MDS.csv")

    # Compare the dataframes
    print("Dataframe shapes:")
    print(f"Old: {df.shape}")
    print(f"New: {df_new.shape}")

    print("\nColumns comparison:")
    print(f"Old columns: {list(df.columns)}")
    print(f"New columns: {list(df_new.columns)}")

    # Check for differences in common columns
    common_columns = set(df.columns) & set(df_new.columns)
    print(f"\nCommon columns: {list(common_columns)}")

    for col in common_columns:
        if not df[col].equals(df_new[col]):
            print(f"\nDifferences in column '{col}':")
            # Find rows where values differ
            mask = df[col] != df_new[col]
            if mask.any():
                print(f"  Rows with differences: {mask.sum()}")
                print("  Old values:")
                print(df.loc[mask, col].head())
                print("  New values:")
                print(df_new.loc[mask, col].head())

    # Check for rows that exist in one but not the other
    if "motif_name" in common_columns:
        old_motifs = set(df["motif_name"])
        new_motifs = set(df_new["motif_name"])

        only_in_old = old_motifs - new_motifs
        only_in_new = new_motifs - old_motifs

        if only_in_old:
            print(f"\nMotifs only in old: {list(only_in_old)}")
        if only_in_new:
            print(f"\nMotifs only in new: {list(only_in_new)}")

    df_new = df_new.query("contains_helix == 1")
    print(df_new["motif_name"].unique())


if __name__ == "__main__":
    main()
