import os
import pandas as pd
import click
import glob
from typing import List

from rna_motif_library.hbond import get_flipped_hbond
from rna_motif_library.motif import get_cached_motifs, get_cached_hbonds
from rna_motif_library.residue import are_residues_connected
from rna_motif_library.util import (
    get_nucleotide_atom_type,
    get_pdb_ids,
    parse_motif_indentifier,
    file_exists_and_has_content,
    get_res_to_motif_id,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)


def are_motifs_connected(motif_1, motif_2):
    """
    Check if two motifs are connected through their strand end residues.

    Args:
        motif_1: First motif to check for connections
        motif_2: Second motif to check for connections

    Returns:
        bool: True if motifs are connected, False otherwise
    """
    # Check each strand's end residues for connections
    for strand1 in motif_1.strands:
        for strand2 in motif_2.strands:
            # Get end residues
            strand1_ends = [strand1[0], strand1[-1]]
            strand2_ends = [strand2[0], strand2[-1]]

            # Check all combinations of end residues
            for res1 in strand1_ends:
                for res2 in strand2_ends:
                    if are_residues_connected(res1, res2) != 0:
                        return True
    return False


def have_common_basepair(motif_1, motif_2):
    """
    Check if two motifs share any common basepairs.

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if motifs share any basepairs, False otherwise
    """
    # Get all basepairs from both motifs
    bps_1 = set((bp.res_1.get_str(), bp.res_2.get_str()) for bp in motif_1.basepairs)
    bps_2 = set((bp.res_1.get_str(), bp.res_2.get_str()) for bp in motif_2.basepairs)

    # Also check reverse order of residues
    bps_1.update((bp.res_2.get_str(), bp.res_1.get_str()) for bp in motif_1.basepairs)
    bps_2.update((bp.res_2.get_str(), bp.res_1.get_str()) for bp in motif_2.basepairs)

    # Check for any common basepairs
    return len(bps_1.intersection(bps_2)) > 0


def are_motifs_sequential(motif_1, motif_2):
    """
    Check if any strands between two motifs are sequential (connected in sequence).

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if any strands are sequential, False otherwise
    """
    # Check each strand's end residues for sequential connections
    for strand1 in motif_1.strands:
        for strand2 in motif_2.strands:
            # Get end residues
            strand1_ends = [strand1[0], strand1[-1]]
            strand2_ends = [strand2[0], strand2[-1]]

            # Check all combinations of end residues
            for res1 in strand1_ends:
                for res2 in strand2_ends:
                    # Check if residues are on same chain
                    if res1.chain_id != res2.chain_id:
                        continue

                    # Check if residue numbers differ by 1
                    if abs(res1.num - res2.num) == 1:
                        return True

    return False


def check_residue_overlap(motif_1, motif_2):
    """
    Check if two motifs have any overlapping residues.

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if motifs have overlapping residues, False otherwise
    """
    # Get all residues from both motifs
    residues_1 = set(res.get_x3dna_str() for res in motif_1.get_residues())
    residues_2 = set(res.get_x3dna_str() for res in motif_2.get_residues())

    # Check for any overlapping residues
    return len(residues_1.intersection(residues_2)) > 0


def find_tertiary_interactions(pdb_id):
    """
    Find tertiary interactions between motifs in a PDB structure.

    Args:
        pdb_id (str): PDB identifier to analyze

    Returns:
        pd.DataFrame: DataFrame containing tertiary interaction data
    """
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {m.name: m for m in motifs}
    motif_res = get_res_to_motif_id(motifs)
    hbonds = get_cached_hbonds(pdb_id)
    data = []
    for hbond in hbonds:
        if not (hbond.res_type_1 == "RNA" and hbond.res_type_2 == "RNA"):
            continue
        motif_1_name = motif_res[hbond.res_1.get_str()]
        motif_2_name = motif_res[hbond.res_2.get_str()]
        if motif_1_name == motif_2_name:
            continue

        # Sort motif names to ensure consistent ordering
        if motif_1_name > motif_2_name:
            hbond = get_flipped_hbond(hbond)
            motif_1_name, motif_2_name = motif_2_name, motif_1_name

        motif_1 = motifs_by_name[motif_1_name]
        motif_2 = motifs_by_name[motif_2_name]
        if are_motifs_connected(motif_1, motif_2):
            continue
        if are_motifs_sequential(motif_1, motif_2):
            continue
        if have_common_basepair(motif_1, motif_2):
            continue
        if check_residue_overlap(motif_1, motif_2):
            continue

        data.append(
            {
                "pdb_id": str(pdb_id),
                "motif_1": motif_1_name,
                "motif_2": motif_2_name,
                "res_1": hbond.res_1.get_str(),
                "res_2": hbond.res_2.get_str(),
                "atom_1": hbond.atom_1,
                "atom_2": hbond.atom_2,
                "atom_type_1": get_nucleotide_atom_type(hbond.atom_1),
                "atom_type_2": get_nucleotide_atom_type(hbond.atom_2),
                "score": hbond.score,
            }
        )
    df = pd.DataFrame(data)
    return df


def write_interactions_to_cif(motifs, dir_name, pos):
    """
    Write motif interactions to CIF files in a specified directory.

    Args:
        motifs (list): List of motif objects to write
        dir_name (str): Base directory name for output
        pos (int): Position identifier for subdirectory

    Returns:
        None
    """
    os.makedirs(os.path.join(dir_name, str(pos)), exist_ok=True)
    for motif in motifs:
        print(motif.name, end=" ")
        motif.to_cif(os.path.join(dir_name, str(pos), f"{motif.name}.cif"))
    print()


def get_duplicate_motifs(pdb_id):
    """
    Get list of duplicate motifs for a specific PDB structure.

    Args:
        pdb_id (str): PDB identifier to check for duplicates

    Returns:
        list: List of duplicate motif names
    """
    path = os.path.join(DATA_PATH, "dataframes", "duplicate_motifs", f"{pdb_id}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df = df[df["is_duplicate"] == True]
    return df["motif"].values


# main functions #######################################################################


def process_pdb_id_for_tc_hbonds(pdb_id):
    """
    Process a single PDB ID to find tertiary contact hydrogen bonds.

    Args:
        pdb_id (str): The PDB ID to process

    Returns:
        pd.DataFrame or None: DataFrame containing tertiary contact hydrogen bonds or None if processing failed
    """
    output_path = os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")

    # If file exists and has content, load and return it
    if file_exists_and_has_content(output_path):
        return pd.read_csv(output_path)

    # Otherwise try to generate new data
    try:
        df_tc = find_tertiary_interactions(pdb_id)
        df_tc["pdb_id"] = pdb_id
        df_tc.to_csv(output_path, index=False)
        return df_tc
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


def process_group(g, pdb_id, unique_motifs):
    """
    Process a group of tertiary contact interactions for analysis.

    Args:
        g (pd.DataFrame): Group of interaction data
        pdb_id (str): PDB identifier
        unique_motifs (list): List of unique motif names

    Returns:
        dict or None: Dictionary containing processed interaction data or None if insufficient data
    """
    if len(g) < 3:
        return None

    hbond_types = {
        "base-base": 0,
        "base-sugar": 0,
        "base-phos": 0,
        "sugar-sugar": 0,
        "phos-sugar": 0,
        "phos-phos": 0,
    }
    motif_1_res = []
    motif_2_res = []

    hbond_score = 0
    for _, row in g.iterrows():
        atom_types = sorted([row["atom_type_1"], row["atom_type_2"]])
        hbond_types[atom_types[0] + "-" + atom_types[1]] += 1
        hbond_score += row["score"]
        if row["res_1"] not in motif_1_res:
            motif_1_res.append(row["res_1"])
        if row["res_2"] not in motif_2_res:
            motif_2_res.append(row["res_2"])

    motif_info_1 = parse_motif_indentifier(g.iloc[0]["motif_1"])
    motif_info_2 = parse_motif_indentifier(g.iloc[0]["motif_2"])
    return {
        "motif_1_id": g.iloc[0]["motif_1"],
        "motif_2_id": g.iloc[0]["motif_2"],
        "motif_1_type": motif_info_1[0],
        "motif_2_type": motif_info_2[0],
        "motif_1_size": motif_info_1[1],
        "motif_2_size": motif_info_2[1],
        "m_sequence_1": motif_info_1[2],
        "m_sequence_2": motif_info_2[2],
        "motif_1_res": motif_1_res,
        "motif_2_res": motif_2_res,
        "num_hbonds": len(g),
        "hbond_score": hbond_score,
        "num_base_base_hbonds": hbond_types["base-base"],
        "num_base_sugar_hbonds": hbond_types["base-sugar"],
        "num_base_phosphate_hbonds": hbond_types["base-phos"],
        "num_phosphate_sugar_hbonds": hbond_types["phos-sugar"],
        "num_phosphate_phosphate_hbonds": hbond_types["phos-phos"],
        "pdb_id": pdb_id,
        "is_motif_1_unique": int(g.iloc[0]["motif_1"] in unique_motifs),
        "is_motif_2_unique": int(g.iloc[0]["motif_2"] in unique_motifs),
    }


def process_pdb_tertiary_contacts(df, pdb_id, unique_motifs):
    """
    Process tertiary contacts for a single PDB structure.

    Args:
        df (pd.DataFrame): DataFrame containing interaction data
        pdb_id (str): PDB identifier
        unique_motifs (list): List of unique motif names

    Returns:
        pd.DataFrame: DataFrame containing processed tertiary contact data
    """
    pdb_data = []
    for _, g in df.groupby(["motif_1", "motif_2"]):
        if len(g) < 3:
            continue
        data = process_group(g, pdb_id, unique_motifs)
        if data is None:
            continue
        pdb_data.append(data)
    df_pdb = pd.DataFrame(pdb_data)
    return df_pdb


def process_pdb_id_for_tertiary_contacts(args):
    """
    Process a single PDB ID to find tertiary contacts. Generates a json file in the
    dataframes/tertiary_contacts directory.

    Args:
        args (tuple): Tuple containing (pdb_id, unique_motifs)

    Returns:
        None
    """
    pdb_id, unique_motifs = args
    if not os.path.exists(
        os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
    ):
        return
    df = pd.read_csv(
        os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
    )
    output_path = os.path.join(
        DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json"
    )
    if len(df) == 0:
        return
    try:
        df_pdbs = process_pdb_tertiary_contacts(df, pdb_id, unique_motifs)
        df_pdbs.to_json(output_path, orient="records")
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return


# main functions #######################################################################


def find_tertiary_contact_hbonds(pdb_ids: List[str], processes: int = 1):
    """
    Find tertiary contact hydrogen bonds for multiple PDB structures using parallel processing.

    Args:
        pdb_ids (List[str]): List of PDB identifiers to process
        processes (int): Number of processes to use for parallel processing (default: 1)

    Returns:
        None
    """
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "tc_hbonds"), exist_ok=True)
    # Process PDB IDs in parallel
    run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_id_for_tc_hbonds,
        processes=processes,
        batch_size=100,
        desc="Processing tertiary contact hydrogen bonds",
    )


def find_tertiary_contacts(
    pdb_ids: List[str], unique_motifs: List[str], processes: int = 1
):
    """
    Find tertiary contacts for multiple PDB structures using parallel processing.

    Args:
        pdb_ids (List[str]): List of PDB identifiers to process
        unique_motifs (List[str]): List of unique motif names
        processes (int): Number of processes to use for parallel processing (default: 1)

    Returns:
        None
    """
    os.makedirs(
        os.path.join(DATA_PATH, "dataframes", "tertiary_contacts"), exist_ok=True
    )
    run_w_processes_in_batches(
        items=[(pdb_id, unique_motifs) for pdb_id in pdb_ids],
        func=process_pdb_id_for_tertiary_contacts,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for tertiary contacts",
    )


def generate_tertiary_contacts_release():
    path = os.path.join(DATA_PATH, "summaries", "tertiary_contacts")
    os.makedirs(path, exist_ok=True)
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", "*.json")
    )
    df = concat_dataframes_from_files(json_files)
    df.to_json(os.path.join(path, "all_tertiary_contacts.json"), orient="records")
    df = df[~((df["is_motif_1_unique"] == False) & (df["is_motif_2_unique"] == False))]
    df.to_json(
        os.path.join(path, "unique_tertiary_contacts.json"),
        orient="records",
    )


# cli ##################################################################################


@click.group()
def cli():
    """
    Command line interface for tertiary contact analysis.
    """
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def run_find_tertiary_contact_hbonds(csv_path, processes):
    """
    Find tertiary contact hydrogen bonds for all PDB IDs using parallel processing.

    Args:
        csv_path (str): Path to CSV file containing PDB IDs
        processes (int): Number of processes to use for parallel processing

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].values
    find_tertiary_contact_hbonds(pdb_ids, processes)


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def run_find_tertiary_contacts(csv_path, processes):
    """
    Find tertiary contacts for all PDB IDs using parallel processing.

    Args:
        csv_path (str): Path to CSV file containing PDB IDs
        processes (int): Number of processes to use for parallel processing

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    df_unique_motifs = pd.read_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    )
    pdb_ids = df["pdb_id"].values
    unique_motifs = list(df_unique_motifs["motif_name"].values)
    find_tertiary_contacts(pdb_ids, unique_motifs, processes)


@cli.command()
def run_generate_tertiary_contacts_release():
    """
    Process and filter tertiary contacts to get only unique motif interactions.
    """
    generate_tertiary_contacts_release()


@cli.command()
def analysis():
    """
    Perform analysis on unique tertiary contacts data.

    Args:
        None

    Returns:
        None
    """
    df = pd.read_json("unique_tertiary_contacts.json")
    print(len(df))
    exit()


if __name__ == "__main__":
    cli()
