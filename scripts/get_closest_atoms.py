import csv
import pandas as pd
from biopandas.mmcif import PandasMmcif
from rna_motif_library.classes import canon_res_list


def main():
    ppdb = PandasMmcif().read_mmcif("data/pdbs/8GWB.cif")
    df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
    # Create lists to store data for CSV
    csv_data = []

    for res in canon_res_list:
        df_sub = df[df["label_comp_id"] == res]
        n_atoms = df_sub[df_sub["label_atom_id"].str.startswith("N")][
            "label_atom_id"
        ].unique()
        o_atoms = df_sub[df_sub["label_atom_id"].str.startswith("O")][
            "label_atom_id"
        ].unique()
        # For each N and O atom in the residue, find its closest neighbor
        if len(df_sub) > 0:
            for atom_id in list(n_atoms) + list(o_atoms):
                # Get coordinates of current atom
                atom = df_sub[df_sub["label_atom_id"] == atom_id].iloc[0]
                x1, y1, z1 = atom["Cartn_x"], atom["Cartn_y"], atom["Cartn_z"]

                # Calculate distances to all other atoms in residue
                other_atoms = df_sub[df_sub["label_atom_id"] != atom_id]
                if len(other_atoms) > 0:
                    distances = (
                        (other_atoms["Cartn_x"] - x1) ** 2
                        + (other_atoms["Cartn_y"] - y1) ** 2
                        + (other_atoms["Cartn_z"] - z1) ** 2
                    ) ** 0.5

                    # Find closest atom
                    min_idx = distances.idxmin()
                    closest_atom = other_atoms.loc[min_idx, "label_atom_id"]
                    min_dist = distances.min()

                    csv_data.append([res, f"{atom_id}", closest_atom])

    with open("closest_atoms.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Residue", "Atom Info", "Value"])
        writer.writerows(csv_data)


if __name__ == "__main__":
    main()
