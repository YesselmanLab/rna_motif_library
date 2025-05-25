import pandas as pd
import glob
import os


def main():
    csv_files = glob.glob("data/dataframes/non_redundant_sets/*.csv")
    for csv_file in csv_files:
        fname = os.path.basename(csv_file)
        old_csv_file = f"data/dataframes/non_redundant_sets_old/{fname}"
        df = pd.read_csv(csv_file)
        old_df = pd.read_csv(old_csv_file)
        # Round RMSD values to 2 decimal places
        df["rmsd"] = df["rmsd"].round(2)
        old_df["rmsd"] = old_df["rmsd"].round(2)

        # Compare if dataframes have same rows
        if df.equals(old_df):
            print(f"{fname}: Dataframes are identical")
        else:
            print(f"{fname}: Dataframes differ")
            # Show differences in row counts
            print(f"New df rows: {len(df)}, Old df rows: {len(old_df)}")
            # Show rows that are in one df but not the other
            new_rows = set(map(tuple, df.values)) - set(map(tuple, old_df.values))
            old_rows = set(map(tuple, old_df.values)) - set(map(tuple, df.values))
            if new_rows:
                print("New rows in current version:")
                print(pd.DataFrame(list(new_rows), columns=df.columns))
            if old_rows:
                print("Rows only in old version:")
                print(pd.DataFrame(list(old_rows), columns=old_df.columns))


main()
