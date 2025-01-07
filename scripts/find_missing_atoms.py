import os
import pandas as pd
import glob
from biopandas.mmcif import PandasMmcif

from rna_motif_library.settings import LIB_PATH
from rna_motif_library.interactions import get_interactions, get_atom_coords
from rna_motif_library.dssr import DSSROutput


def process_pdb(pdb_path):
    ppdb = PandasMmcif().read_mmcif(pdb_path)
    df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
    name = os.path.basename(pdb_path)[:-4]
    json_path = os.path.join(LIB_PATH, "data/dssr_output", f"{name}.json")
    d_out = DSSROutput(json_path=json_path)
    hbonds = d_out.get_hbonds()
    interactions = get_interactions(name, hbonds)

    results = []
    for interaction in interactions:
        coords_1 = get_atom_coords(
            interaction.atom_1,
            interaction.res_1.res_id,
            interaction.res_1.chain_id,
            df,
        )
        coords_2 = get_atom_coords(
            interaction.atom_2,
            interaction.res_2.res_id,
            interaction.res_2.chain_id,
            df,
        )
        if coords_1 is None:
            results.append(
                f"{name}, {interaction.atom_1}, {interaction.res_1.res_id}, {interaction.res_1.chain_id}\n"
            )
        if coords_2 is None:
            results.append(
                f"{name}, {interaction.atom_2}, {interaction.res_2.res_id}, {interaction.res_2.chain_id}\n"
            )

    print(f"Processed {name}")
    return results


def main():
    RESOURCE_PATH = os.path.join(LIB_PATH, "rna_motif_library", "resources")
    f = os.path.join(RESOURCE_PATH, "closest_atoms.csv")
    df = pd.read_csv(f)
    closest_atoms = {}
    for _, row in df.iterrows():
        closest_atoms[row["residue"] + "-" + row["atom_1"]] = row["atom_2"]

    pdb_files = glob.glob(os.path.join(LIB_PATH, "data", "pdbs", "*.cif"))

    from multiprocessing import Pool

    num_cores = 20  # Specify number of cores to use
    with Pool(processes=num_cores) as pool:
        all_results = pool.map(process_pdb, pdb_files)

    # Flatten results list
    missing_atoms = [item for sublist in all_results for item in sublist]

    with open("missing_atoms.txt", "w") as f:
        for line in missing_atoms:
            f.write(line)


if __name__ == "__main__":
    main()
