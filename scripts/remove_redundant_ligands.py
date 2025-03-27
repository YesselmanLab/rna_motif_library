import pandas as pd 

def main():
    df = pd.read_json("rna_ligand_interactions.json")
    for i, g in df.groupby("pdb_id"):
        if len(g) == 2:
            print(g)

if __name__ == "__main__":
    main()
