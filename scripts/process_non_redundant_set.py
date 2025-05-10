import click
import pandas as pd

@click.group()
def cli():
    pass

@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def get_pdb_ids(csv_path):
    df = pd.read_csv(csv_path, header=None)
    data = []
    for _, row in df.iterrows():
        row = row.tolist()
        spl = row[1].split("|")
        if spl[0] not in data:
            data.append(spl[0])
    df = pd.DataFrame(data, columns=["pdb_id"])
    df.to_csv("non_redundant_set.csv", index=False)

@cli.command()
def compare_sets():
    df = pd.read_csv("non_redundant_set.csv")
    df2 = pd.read_csv("rna_structures.txt", sep=",")
    
    # Get lists of PDB IDs from both dataframes
    df_pdbs = set(df["pdb_id"].values)
    df2_pdbs = set(df2["PDB_ID"].values)
    
    # Find PDBs in df that are not in df2
    missing_pdbs = df_pdbs - df2_pdbs
    
    if missing_pdbs:
        print(f"Found {len(missing_pdbs)} PDBs in non_redundant_set.csv that are not in rna_structures.csv:")
        for pdb in missing_pdbs:
            print(pdb)
    else:
        print("All PDBs in non_redundant_set.csv are present in rna_structures.csv")


if __name__ == "__main__":
    cli()
