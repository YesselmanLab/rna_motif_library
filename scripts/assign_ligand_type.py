import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcNumAromaticRings

from rna_motif_library.ligand import DATA_PATH

# pd.set_option("future.no_silent_downcasting", True)


def read_sdf_file(file_path):
    """Reads an SDF file and returns a list of RDKit molecule objects.

    Args:
        file_path: The path to the SDF file.

    Returns:
        A list of RDKit molecule objects, or an empty list if an error occurs.
    """
    try:
        suppl = Chem.SDMolSupplier(file_path)
        molecules = [mol for mol in suppl if mol is not None]
        return molecules
    except Exception as e:
        print(f"Error reading SDF file: {e}")
        return []


def check_multiple_assignments(df):
    """
    Check for rows with multiple type assignments
    """
    # Check for rows with multiple True assignments
    multiple_assignments = df[
        (df["assigned_ligand"] & df["assigned_solvent"])
        | (df["assigned_ligand"] & df["assigned_polymer"])
        | (df["assigned_solvent"] & df["assigned_polymer"])
    ]

    if len(multiple_assignments) > 0:
        print("WARNING: Found rows with multiple assignments:")
        for _, row in multiple_assignments.iterrows():
            print(row)
            exit()
    exit()


def assign_ligand_type():
    """
    Assign ligand type to each row in the dataframe
    """
    df = pd.read_json(os.path.join(DATA_PATH, "ligands", "ligand_info_final.json"))
    df.rename(columns={"type": "pdb_type"}, inplace=True)
    df_lig = pd.read_csv("likely_ligands.csv")
    df_lig["assigned_ligand"] = True
    df_solv = pd.read_csv("solvent_and_buffers.csv")
    df_solv["assigned_solvent"] = True
    df_poly = pd.read_csv("likely_polymer.csv")
    df_poly["assigned_polymer"] = True
    df["type"] = ""
    df = df.merge(df_lig, on="id", how="left")
    df = df.merge(df_solv, on="id", how="left")
    df = df.merge(df_poly, on="id", how="left")
    df["assigned_ligand"] = (
        df["assigned_ligand"].fillna(False).infer_objects(copy=False)
    )
    df["assigned_solvent"] = (
        df["assigned_solvent"].fillna(False).infer_objects(copy=False)
    )
    df["assigned_polymer"] = (
        df["assigned_polymer"].fillna(False).infer_objects(copy=False)
    )
    for i, row in df.iterrows():
        noncovalent_count = len(row["noncovalent_results"])
        covalent_count = len(row["covalent_results"])
        polymer_count = len(row["polymer_results"])
        if noncovalent_count == 0 and covalent_count == 0 and polymer_count > 1:
            df.at[i, "type"] = "polymer"
        elif noncovalent_count > 0 and covalent_count == 0 and polymer_count == 0:
            df.at[i, "type"] = "small-molecule"
        elif noncovalent_count == 0 and covalent_count > 1 and polymer_count == 0:
            df.at[i, "type"] = "covalent"
        else:
            df.at[i, "type"] = "unknown"
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_final_w_types.json"),
        orient="records",
    )


def assign_ligand_type_features():
    """
    Assign ligand type features to each row in the dataframe
    """
    df = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_final_w_types.json")
    )
    df["aromatic_rings"] = 0
    df["h_acceptors"] = 0
    df["h_donors"] = 0
    for i, row in df.iterrows():
        try:
            mol = read_sdf_file(
                os.path.join(DATA_PATH, "residues_w_h_sdfs", f"{row['id']}_ideal.sdf")
            )[0]
            df.at[i, "aromatic_rings"] = CalcNumAromaticRings(mol)
            df.at[i, "h_acceptors"] = CalcNumHBA(mol)
            df.at[i, "h_donors"] = CalcNumHBD(mol)
        except Exception as e:
            df.at[i, "aromatic_rings"] = -1
            df.at[i, "h_acceptors"] = -1
            df.at[i, "h_donors"] = -1
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_final_w_types_features.json"),
        orient="records",
    )


def main():
    """
    main function for script
    """
    # manually edited to assign ligands with no aromatic rings to small-molecule
    df = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_final_w_types_features.json")
    )
    df_sub = df[df["type"] == "small-molecule"]
    # probably a ligand with aromatic rings
    df_sub = df_sub[df_sub["aromatic_rings"] > 0]
    df_sub["assigned_ligand"] = True
    df_unassigned = df[
        (df["assigned_ligand"] == False)
        & (df["assigned_solvent"] == False)
        & (df["assigned_polymer"] == False)
    ]
    count = 0
    df_unassigned.sort_values(by="formula_weight", ascending=True, inplace=True)
    for i, row in df_unassigned.iterrows():
        noncovalent_count = len(row["noncovalent_results"])
        covalent_count = len(row["covalent_results"])
        polymer_count = len(row["polymer_results"])
        if noncovalent_count == 0 and covalent_count == 0 and polymer_count > 1:
            df.at[i, "assigned_polymer"] = True
        elif noncovalent_count > 1 and covalent_count == 0 and polymer_count == 0:
            df.at[i, "assigned_ligand"] = True
            count += 1
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_complete_final.json"),
        orient="records",
        indent=4,
    )
    print(len(df_unassigned))
    print(count)


if __name__ == "__main__":
    main()
