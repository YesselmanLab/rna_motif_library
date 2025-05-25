import os
import pandas as pd
import click
import glob

from rna_motif_library.motif import get_cached_motifs, get_cached_hbonds
from rna_motif_library.residue import are_residues_connected
from rna_motif_library.util import (
    get_nucleotide_atom_type,
    get_pdb_ids,
    parse_motif_name,
    file_exists_and_has_content,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)


def get_non_redundant_pdb_ids():
    df = pd.read_csv("data/csvs/non_redundant_set.csv")
    pdb_ids = df["pdb_id"].tolist()
    return pdb_ids


def get_unique_res(pdb_id, motifs):
    path = os.path.join(DATA_PATH, "dataframes", "duplicate_motifs", f"{pdb_id}.csv")
    dup_motifs = []
    unique_res = []
    if os.path.exists(path):
        try:
            df_dup = pd.read_csv(path)
            dup_motifs = df_dup["dup_motif"].values
        except Exception as e:
            dup_motifs = []

    for m in motifs:
        if m.name in dup_motifs:
            continue
        for r in m.get_residues():
            if r.get_str() not in unique_res:
                unique_res.append(r.get_str())
    return unique_res


def are_motifs_connected(motif_1, motif_2):
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
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {m.name: m for m in motifs}
    motif_res = {}
    motif_res_pairs = {}
    for motif in motifs:
        for res in motif.get_residues():
            motif_res[res.get_str()] = motif.name
            motif_res_pairs[motif.name + "-" + res.get_str()] = True
    hbonds = get_cached_hbonds(pdb_id)
    data = []
    for hbond in hbonds:
        if not (hbond.res_type_1 == "RNA" and hbond.res_type_2 == "RNA"):
            continue
        motif_1_name = motif_res[hbond.res_1.get_str()]
        motif_2_name = motif_res[hbond.res_2.get_str()]
        if motif_1_name == motif_2_name:
            continue
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
    os.makedirs(os.path.join(dir_name, str(pos)), exist_ok=True)
    for motif in motifs:
        print(motif.name, end=" ")
        motif.to_cif(os.path.join(dir_name, str(pos), f"{motif.name}.cif"))
    print()


def get_duplicate_motifs(pdb_id):
    path = os.path.join(DATA_PATH, "dataframes", "duplicate_motifs", f"{pdb_id}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df = df[df["is_duplicate"] == True]
    return df["motif"].values


# main functions #######################################################################


def process_pdb_id_for_tc_hbonds(pdb_id):
    """Process a single PDB ID to find tertiary contact hydrogen bonds.

    Args:
        pdb_id: The PDB ID to process

    Returns:
        DataFrame containing tertiary contact hydrogen bonds or None if processing failed
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

    # if (
    #    g.iloc[0]["motif_1"] not in unique_motifs
    #    and g.iloc[0]["motif_2"] not in unique_motifs
    # ):
    #    return None

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
        motif_1_res.append(row["res_1"])
        motif_2_res.append(row["res_2"])

    motif_info_1 = parse_motif_name(g.iloc[0]["motif_1"])
    motif_info_2 = parse_motif_name(g.iloc[0]["motif_2"])
    return {
        "motif_1": g.iloc[0]["motif_1"],
        "motif_2": g.iloc[0]["motif_2"],
        "mtype_1": motif_info_1[0],
        "mtype_2": motif_info_2[0],
        "m_size_1": motif_info_1[1],
        "m_size_2": motif_info_2[1],
        "m_sequence_1": motif_info_1[2],
        "m_sequence_2": motif_info_2[2],
        "motif_1_res": motif_1_res,
        "motif_2_res": motif_2_res,
        "pdb_id": motif_info_1[3],
        "num_hbonds": len(g),
        "hbond_score": hbond_score,
        "base-base": hbond_types["base-base"],
        "base-sugar": hbond_types["base-sugar"],
        "base-phos": hbond_types["base-phos"],
        "phos-sugar": hbond_types["phos-sugar"],
        "phos-phos": hbond_types["phos-phos"],
        "pdb_id": pdb_id,
        "unique_motif_1": g.iloc[0]["motif_1"] in unique_motifs,
        "unique_motif_2": g.iloc[0]["motif_2"] in unique_motifs,
    }


def process_pdb_tertiary_contacts(df, pdb_id, unique_motifs):
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
    """Process a single PDB ID to find tertiary contacts.

    Args:
        pdb_id: The PDB ID to process
        df: DataFrame containing all tertiary contact hydrogen bonds
        unique_motifs: List of unique motif names

    Returns:
        DataFrame containing tertiary contacts or None if processing failed
    """
    pdb_id, df, unique_motifs = args
    output_path = os.path.join(
        DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json"
    )

    # If file exists and has content, load and return it
    if file_exists_and_has_content(output_path):
        return pd.read_json(output_path)

    # Otherwise try to generate new data
    try:
        g = df[df["pdb_id"] == pdb_id]
        df_pdbs = process_pdb_tertiary_contacts(g, pdb_id, unique_motifs)
        df_pdbs.to_json(output_path, orient="records")
        return df_pdbs
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


# cli ##################################################################################


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def find_tc_hbonds(processes):
    """Find tertiary contact hydrogen bonds for all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    pdb_ids = get_pdb_ids()

    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_id_for_tc_hbonds,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for tertiary contact hydrogen bonds",
    )

    # Collect results from processed files
    dfs = []
    for result in results:
        if result is not None:
            dfs.append(result)

    # Combine all results
    if dfs:
        df = pd.concat(dfs)
        df["pdb_id"] = df["pdb_id"].astype(str)
        df.to_csv("tertiary_contacts_hbonds.csv", index=False)


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def find_tertiary_contacts(processes):
    """Find tertiary contacts for all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    df = pd.read_csv("tertiary_contacts_hbonds.csv", dtype={"pdb_id": str})
    path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv")
    unique_motifs = list(pd.read_csv(path)["motif_name"].values)

    # Get unique PDB IDs
    pdb_ids = df["pdb_id"].unique().tolist()

    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=[(pdb_id, df, unique_motifs) for pdb_id in pdb_ids],
        func=process_pdb_id_for_tertiary_contacts,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for tertiary contacts",
    )

    # Collect results from processed files
    dfs = []
    for result in results:
        if result is not None:
            dfs.append(result)

    # Combine all results if needed
    if dfs:
        df_combined = pd.concat(dfs)
        df_combined.to_json("all_tertiary_contacts.json", orient="records")


@cli.command()
def get_unique_tertiary_contacts():
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", "*.json")
    )
    df = concat_dataframes_from_files(json_files)
    df = df[~((df["unique_motif_1"] == False) & (df["unique_motif_2"] == False))]
    df.to_json("unique_tertiary_contacts.json", orient="records")


@cli.command()
def analysis():
    pass


if __name__ == "__main__":
    cli()
