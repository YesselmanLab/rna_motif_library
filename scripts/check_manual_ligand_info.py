import os
import pandas as pd
from rna_motif_library.settings import VERSION, DATA_PATH

LIGAND_DATA_PATH = os.path.join(DATA_PATH, "ligands")


def find_and_update_assignments(df, df_manual, assigned_col):
    # Get all IDs marked with the assigned column in main ligand info
    assigned_ids = df[df[assigned_col]]["id"].values

    # Get all IDs from manual list
    manual_ids = df_manual["id"].values

    # Find IDs marked as assigned but missing from manual list
    missing_ids = set(assigned_ids) - set(manual_ids)

    if len(missing_ids) > 0:
        print(f"Found residues marked as {assigned_col} but missing from manual list:")
        # Create records for new assignments
        new_records = []
        for res_id in missing_ids:
            res_info = df[df["id"] == res_id].iloc[0]
            print(f"ID: {res_id}, Name: {res_info['name']}")
            new_records.append({"id": res_id})

        # Create updated dataframe with just IDs
        df_new = pd.DataFrame(new_records)
        df_updated = pd.concat([df_manual[["id"]], df_new], ignore_index=True)

        # Save updated CSV
        df_updated.to_csv("test.csv", index=False)
    else:
        print(f"All {assigned_col} assignments match manual list")


def main():
    df = pd.read_json(
        os.path.join(
            LIGAND_DATA_PATH,
            "summary",
            "versions",
            f"v{VERSION}",
            "ligand_info.json",
        )
    )
    df_solvent = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "solvent_and_buffers.csv")
    )
    df_ligands = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "ligands.csv")
    )
    df_poly = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "polymers.csv")
    )

    # find_and_update_assignments(df, df_solvent, "assigned_solvent")
    # find_and_update_assignments(df, df_ligands, "assigned_ligand")
    find_and_update_assignments(df, df_poly, "assigned_polymer")


if __name__ == "__main__":
    main()
