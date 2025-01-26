import time
import warnings
import os
import click
import sys
import functools
import pandas as pd


from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.logger import setup_logging, get_logger

from rna_motif_library.basepair import (
    generate_basepairs,
    save_basepairs_to_json,
)
from rna_motif_library.hbond import generate_hbonds_from_x3dna, save_hbonds_to_json
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
    "--skip_existing",
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
        print(pdb_id)
        if skip_existing and os.path.exists(get_cached_path(pdb_id, "residues")):
            log.info(f"Skipping {pdb_id} because it already exists")
            continue
        df_atoms = pd.read_parquet(
            os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_id}.parquet")
        )
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
        residues = get_cached_residues(pdb_id)
        chains = get_rna_chains(list(residues.values()))
        for i, chain in enumerate(chains):
            write_chain_to_cif(chain, f"{pdb_id}_{i}.cif")
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
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.option("--overwrite", is_flag=True, help="Overwrite existing interactions")
@time_func
def process_interactions(pdb, directory, debug, overwrite):
    """
    Processes interactions from source PDB using data from DSSR and interactions using data from SNAP.
    """
    setup_logging(debug=debug)
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(DATA_PATH, "jsons", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "jsons", "basepairs"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "hbonds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "basepairs"), exist_ok=True)
    pdb_ids = get_pdb_ids(pdb, directory)
    log.info(f"Processing {len(pdb_ids)} PDBs")
    for pdb_id in pdb_ids:
        # TODO fill in later
        hbonds = generate_hbonds_from_x3dna(pdb_id)
        dssr_output = get_cached_dssr_output(pdb_id)
        dssr_pairs = dssr_output.get_pairs()
        residues = get_cached_residues(pdb_id)
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
@click.option("--debug", is_flag=True, help="Run in debug mode")
@time_func
def generate_motifs(pdb, directory, debug):
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
    pdb_ids = get_pdb_ids(pdb, directory)
    generate_motif_files(pdb_ids)


if __name__ == "__main__":
    cli()
