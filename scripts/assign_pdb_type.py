import pandas as pd


def find_ribosome_rows(df):
    """
    Find all rows that have 'ribosome' in their title (case insensitive) and set 'is_ribosome' column to True.
    Only includes rows where count is greater than 100.

    Args:
        df (pd.DataFrame): DataFrame containing a 'title' column and 'count' column

    Returns:
        pd.DataFrame: DataFrame with new 'is_ribosome' column set to True for matching rows
    """
    s = "ribosom|disome|trisome|70S|80S|48S|30S|43S"
    df_copy = df.copy()
    df_copy["is_ribosome"] = (
        df_copy["title"].str.contains(s, case=False, na=False)
        | df_copy["pdbx_keywords"].str.contains(s, case=False, na=False)
        | df_copy["other_keywords"].str.contains(s, case=False, na=False)
    ) & (df_copy["count"] > 400)
    return df_copy


def main():
    df = pd.read_json("pdb_titles.json")
    also_ribosome = ["6X6T", "7OBR", "8BF9"]
    df_rna_count = pd.read_csv("data/csvs/rna_residue_counts.csv")
    df = df.merge(df_rna_count, on="pdb_id")
    # Find and display ribosome-related entries
    df = find_ribosome_rows(df)
    df.loc[df["pdb_id"].isin(also_ribosome), "is_ribosome"] = True
    df.sort_values(by="count", ascending=False, inplace=True)
    df.to_json("data/summaries/pdb_info.json", orient="records")


if __name__ == "__main__":
    main()
