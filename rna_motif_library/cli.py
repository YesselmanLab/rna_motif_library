import time
import glob
import warnings
import os
import click
import sys
import functools
import pandas as pd
import multiprocessing
from biopandas.mmcif import PandasMmcif


from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.logger import setup_logging, get_logger

from rna_motif_library.basepair import (
    generate_basepairs,
    save_basepairs_to_json,
)
from rna_motif_library.hbond import generate_hbonds, save_hbonds_to_json
from rna_motif_library.residue import (
    Residue,
    save_residues_to_json,
    get_cached_residues,
)
from rna_motif_library.update_library import (
    get_dssr_files,
    get_snap_files,
    download_cif_files,
    generate_motif_files,
)
from rna_motif_library.chain import (
    get_rna_chains,
    get_protein_chains,
    save_chains_to_json,
    write_chain_to_cif,
)
from rna_motif_library.util import (
    get_pdb_ids,
    sanitize_x3dna_atom_name,
    get_x3dna_res_id,
    get_cached_path,
)
from rna_motif_library.x3dna import (
    X3DNAResidueFactory,
    get_cached_dssr_output,
    get_residue_type,
)


log = get_logger("cli")


def time_func(func):
    """
    Decorator to set up logging and log the start and end of the function execution.

    Args:
        func (callable): The function to wrap.

    Returns:
        callable: The wrapped function with logging and setup.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "debug" not in kwargs:
            kwargs["debug"] = False
        func_name = func.__name__
        log.info("Ran at commandline as: %s", " ".join(sys.argv))
        start_time = time.time()
        log.info("Starting time: %s" % pd.Timestamp.now())
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            log.info(
                f"{func_name} execution completed. Total runtime: {formatted_time}"
            )
        return result

    return wrapper


def get_new_pdb_ids():
    df = pd.read_csv("/Users/jyesselman2/Documents/new_pdb_list.txt")
    pdb_ids = df["pdb_id"].tolist()
    keep_ids = []
    for pdb_id in pdb_ids:
        residues = get_cached_residues(pdb_id)
        if len(residues) < 1000:
            keep_ids.append(pdb_id)
    return keep_ids


def get_non_redundant_pdb_ids():
    df = pd.read_csv("data/csvs/non_redundant_set.csv")
    df_count = pd.read_csv("rna_residue_counts.csv")
    count = {row["pdb_id"]: row["count"] for _, row in df_count.iterrows()}
    pdb_ids = df["pdb_id"].tolist()
    final_pdb_ids = []
    for pdb_id in pdb_ids:
        if pdb_id not in count:
            continue
        if count[pdb_id] < 500:
            final_pdb_ids.append(pdb_id)
    return final_pdb_ids


def process_cif(cif_file):
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@time_func
def download_cifs(csv_path, threads, debug):
    """
    Downloads CIFs specified in the CSV from the RCSB PDB database.

    Args:
        threads (int): Number of threads to run on.
        debug (bool): Enable debugging output.

    Returns:
        None

    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    download_cif_files(csv_path, threads)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--directory", default=None, help="Directory to PDBs used")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@time_func
def process_dssr(threads, directory, debug):
    """
    Processes every downloaded PDB with DSSR, extracting the secondary structure into a JSON.

    Args:
        threads (int): Number of threads to run on.
        directory (str): Directory to use for processing.
        Directory = "/notebooks/distributed_sets/set_i/"

    Returns:
        None

    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    get_dssr_files(threads, directory)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--directory", default=None, help="Directory of PDBs to process")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@time_func
def process_snap(threads, directory, debug):
    """
    Processes every downloaded PDB with SNAP, extracting RNA-protein interaction data.

    Args:
        threads (int): Number of threads to run on.

    Returns:
        None

    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    get_snap_files(threads, directory)


@cli.command()
@click.option("--threads", default=1, help="Number of threads to use.")
@click.option("--debug", is_flag=True, help="Enable debugging.")
@time_func
def get_pdb_dfs(threads, debug):
    glob_path = os.path.join("data/pdbs", "*.cif")
    cif_files = glob.glob(glob_path)
    # Filter out files that already have parquet output
    filtered_cif_files = []
    for cif_file in cif_files:
        pdb_name = os.path.basename(cif_file).split(".")[0]
        parquet_path = f"data/pdbs_dfs/{pdb_name}.parquet"
        if not os.path.exists(parquet_path):
            filtered_cif_files.append(cif_file)

    print(f"Found {len(cif_files)} total CIF files")
    print(f"Processing {len(filtered_cif_files)} files that need conversion")
    if len(filtered_cif_files) == 0:
        print("No files need conversion")
        return

    cif_files = filtered_cif_files

    # Split files into chunks for parallel processing
    num_processes = 8  # Reduced from 20 to lower system load
    chunk_size = len(cif_files) // num_processes
    chunks = [
        cif_files[i : i + chunk_size] for i in range(0, len(cif_files), chunk_size)
    ]

    # Create pool and run processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_chunk, chunks)


@cli.command()
@click.argument("n_splits", type=int)
def generate_splits(n_splits):
    os.makedirs("splits", exist_ok=True)
    pdb_ids = get_pdb_ids()
    # Create n_splits CSV files containing evenly distributed PDB IDs
    split_size = len(pdb_ids) // n_splits
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_splits - 1 else len(pdb_ids)
        split_pdb_ids = pdb_ids[start_idx:end_idx]

        df = pd.DataFrame({"pdb_id": split_pdb_ids})
        df.to_csv(f"splits/split_{i}.csv", index=False)


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option(
    "-se",
    "--skip-existing",
    is_flag=True,
    help="Skip existing residues",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@time_func
def process_residues(pdb, directory, debug, skip_existing):
    """
    Processes residues from source PDB using data from DSSR.
    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "residues"), exist_ok=True)
    pdb_ids = get_pdb_ids(pdb, directory)
    log.info(f"Processing {len(pdb_ids)} PDBs")
    for pdb_id in pdb_ids:
        if skip_existing and os.path.exists(get_cached_path(pdb_id, "residues")):
            log.info(f"Skipping {pdb_id} because it already exists")
            continue
        df_atoms = pd.read_parquet(
            os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_id}.parquet")
        )
        print(pdb_id)
        residues = {}
        for i, g in df_atoms.groupby(
            ["auth_asym_id", "auth_seq_id", "auth_comp_id", "pdbx_PDB_ins_code"]
        ):
            coords = g[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            atom_names = g["auth_atom_id"].tolist()
            atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
            chain_id, res_num, res_name, ins_code = i
            if ins_code == "None" or ins_code is None:
                ins_code = ""
            res = Residue(
                chain_id,
                res_name,
                int(res_num),
                ins_code,
                get_residue_type(res_name),
                atom_names,
                coords,
            )
            residues[res.get_str()] = res
        # Save residues to json file
        save_residues_to_json(residues, get_cached_path(pdb_id, "residues"))


@cli.command()
@click.option("--pdb", default=None, type=str, help="Process a specific PDB")
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@time_func
def generate_chains(pdb, directory, debug):
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "chains"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "jsons", "protein_chains"), exist_ok=True)
    pdb_ids = get_pdb_ids(pdb, directory)
    for pdb_id in pdb_ids:
        if os.path.exists(get_cached_path(pdb_id, "chains")):
            log.info(f"Skipping {pdb_id} because it already exists")
            continue
        residues = get_cached_residues(pdb_id)
        chains = get_rna_chains(list(residues.values()))
        # for i, chain in enumerate(chains):
        #    write_chain_to_cif(chain, f"{pdb_id}_{i}.cif")
        save_chains_to_json(chains, get_cached_path(pdb_id, "chains"))
        chains = get_protein_chains(list(residues.values()))
        save_chains_to_json(chains, get_cached_path(pdb_id, "protein_chains"))


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--csv", default=None, type=str, help="CSV file with PDB IDs")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.option("--overwrite", is_flag=True, help="Overwrite existing interactions")
@time_func
def process_interactions(pdb, directory, debug, overwrite, csv):
    """
    Processes interactions from source PDB using data from DSSR and interactions using data from SNAP.
    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "jsons", "basepairs"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "basepairs"), exist_ok=True)
    pdb_ids = get_pdb_ids(pdb, directory, csv_path=csv)
    count = 0
    log.info(f"Processing {len(pdb_ids)} PDBs")
    for pdb_id in pdb_ids:
        count += 1
        print(pdb_id, count)
        hbonds = generate_hbonds(pdb_id)
        residues = get_cached_residues(pdb_id)
        dssr_output = get_cached_dssr_output(pdb_id)
        dssr_pairs = dssr_output.get_pairs()
        basepairs = generate_basepairs(pdb_id, dssr_pairs, residues)
        log.info(
            f"Processed {pdb_id} with {len(hbonds)} hbonds and {len(basepairs)} basepairs"
        )
        save_hbonds_to_json(hbonds, get_cached_path(pdb_id, "hbonds"))
        save_basepairs_to_json(basepairs, get_cached_path(pdb_id, "basepairs"))


@cli.command()
@click.option(
    "--pdb",
    default=None,
    type=str,
    help="Process a specific PDB within the set (defaults to all).",
)
@click.option(
    "--directory",
    default=None,
    type=str,
    help="The directory where the PDBs are located",
)
@click.option("--csv", default=None, type=str, help="CSV file with PDB IDs")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@time_func
def generate_motifs(pdb, directory, debug, csv):
    """
    Extracts motifs from source PDB using data from DSSR, and interactions using data from DSSR and SNAP.

    Args:
        limit (int): Number of PDBs to process (defaults to all).
        pdb (str): Specific PDB ID to process (all by default).
        threads (int): number of threads to use (default: 1)

    Returns:
        None

    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    pdb_ids = get_pdb_ids(pdb, directory, csv_path=csv)
    generate_motif_files(pdb_ids)


if __name__ == "__main__":
    cli()
