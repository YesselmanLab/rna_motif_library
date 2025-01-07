import pandas as pd
import glob
import os
from biopandas.mmcif import PandasMmcif
import multiprocessing


def dataframe_to_cif(df: pd.DataFrame, file_path: str) -> None:
    with open(file_path, "w") as f:
        # Write the CIF header section
        f.write("data_\n")
        f.write("_entry.id " + file_path.split(".")[0] + "\n")
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.auth_atom_id\n")
        f.write("_atom_site.auth_comp_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")

        # Write the data from the DataFrame
        for _, row in df.iterrows():
            f.write(
                "{:<8}{:<7}{:<6}{:<6}{:<6}{:<6}{:<6}{:<12}{:<12}{:<12}\n".format(
                    str(row["group_PDB"]),
                    str(row["id"]),
                    str(row["auth_atom_id"]),
                    str(row["auth_comp_id"]),
                    str(row["auth_asym_id"]),
                    str(row["auth_seq_id"]),
                    str(row["pdbx_PDB_ins_code"]),
                    str(row["Cartn_x"]),
                    str(row["Cartn_y"]),
                    str(row["Cartn_z"]),
                )
            )


cols = [
    "group_PDB",
    "id",
    "auth_atom_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
]


def process_cif(cif_file):
    pdb_name = os.path.basename(cif_file).split(".")[0]
    try:
        ppdb = PandasMmcif().read_mmcif(cif_file)
        df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
        df = df[cols]
        # Write with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df.to_parquet(f"data/pdbs_dfs/{pdb_name}.parquet")
                break
            except TimeoutError:
                if attempt == max_retries - 1:
                    print(f"Failed to write {pdb_name} after {max_retries} attempts")
                    raise
                print(f"Timeout writing {pdb_name}, retrying...")
                time.sleep(1)  # Wait before retry
        return pdb_name
    except Exception as e:
        print(f"Error processing {pdb_name}: {str(e)}")
        return None


def process_chunk(cif_files):
    for cif_file in cif_files:
        try:
            pdb_name = process_cif(cif_file)
            if pdb_name:
                print(f"Processed: {pdb_name}")
        except Exception as e:
            print(f"Failed to process {cif_file}: {str(e)}")


def main():
    glob_path = os.path.join("data/pdbs", "*.cif")
    cif_files = glob.glob(glob_path)

    # Split files into chunks for parallel processing
    num_processes = 8  # Reduced from 20 to lower system load
    chunk_size = len(cif_files) // num_processes
    chunks = [
        cif_files[i : i + chunk_size] for i in range(0, len(cif_files), chunk_size)
    ]

    # Create pool and run processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_chunk, chunks)


if __name__ == "__main__":
    main()
