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
    parse_residue_identifier,
    wc_basepairs_w_gu,
)
from rna_motif_library.dataframe_tools import add_motif_indentifier_columns
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

# main functions #######################################################################

def process_pdb_id_for_tc_hbonds(pdb_id):
    """
    Find tertiary interactions between motifs in a PDB structure.

    Args:
        pdb_id (str): PDB identifier to analyze

    Returns:
        pd.DataFrame: DataFrame containing tertiary interaction data
    """
    output_path = os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
    if file_exists_and_has_content(output_path):
        return None

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
    df.to_csv(output_path, index=False)
    return df


def process_pdb_id_for_tc_basepairs(pdb_id):
    """
    Find tertiary interactions between motifs in a PDB structure.
    """
    output_path = os.path.join(DATA_PATH, "dataframes", "tc_basepairs", f"{pdb_id}.csv")
    if file_exists_and_has_content(output_path):
        return None
    path = os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
    if not os.path.exists(path):
        return None
    df_hbonds = pd.read_csv(path)
    tc_hbonds = {}
    for i, row in df_hbonds.iterrows():
        sorted_res = sorted([row["res_1"], row["res_2"]])
        motifs = [row["motif_1"], row["motif_2"]]
        if sorted_res[0] != row["res_1"]:
            motifs = motifs[::-1]
        tc_hbonds[sorted_res[0] + "-" + sorted_res[1]] = motifs
    df = pd.read_json(
        os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_id}.json")
    )
    data = []
    for _, row in df.iterrows():
        sorted_res = sorted([row["res_1"], row["res_2"]])
        if sorted_res[0] + "-" + sorted_res[1] not in tc_hbonds:
            continue
        motifs = tc_hbonds[sorted_res[0] + "-" + sorted_res[1]]
        data.append(
            [
                sorted_res[0],
                sorted_res[1],
                motifs[0],
                motifs[1],
                row["lw"],
                row["hbond_score"],
                pdb_id
            ]
        ) 
    df = pd.DataFrame(
        data, columns=["res_1", "res_2", "motif_1", "motif_2", "lw", "hbond_score", "pdb_id"]
    )
    df.to_csv(output_path, index=False)
    return df


class TertiaryContactProcessor:
    def __init__(self):
        pass 

    def setup(self, pdb_id, unique_motifs):
        self.pdb_id = pdb_id
        self.motifs = get_cached_motifs(pdb_id)
        self.motifs_by_name = {m.name: m for m in self.motifs}
        self.unique_motifs = unique_motifs
        path = os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
        try:
            self.df_hbonds = pd.read_csv(path)
        except:
            self.df_hbonds = pd.DataFrame()
        path = os.path.join(DATA_PATH, "dataframes", "tc_basepairs", f"{pdb_id}.csv")
        if os.path.exists(path):
            self.df_basepairs = pd.read_csv(path)
        else:
            self.df_basepairs = pd.DataFrame(
                columns=["res_1", "res_2", "motif_1", "motif_2", "lw", "hbond_score"]
            )

    def process(self):
        if len(self.df_hbonds) == 0:
            return pd.DataFrame()
        all_data = []
        for (motif_1, motif_2), g in self.df_hbonds.groupby(["motif_1", "motif_2"]):
            data = self._process_single_tertiary_contact(motif_1, motif_2, g)
            all_data.append(data)
        df = pd.DataFrame(all_data)
        return df

    def _process_single_tertiary_contact(self, motif_1, motif_2, df_hbonds):
        data = {
            "pdb_id": self.pdb_id
        }
        data.update(self._get_motif_info(motif_1, motif_2)) # add motif info to total data
        data.update(self._get_hbond_info(df_hbonds)) # add hbond to total data
        data.update(self._get_residue_info(df_hbonds)) # add residue info to total data
        data.update(self._get_basepair_info(motif_1, motif_2)) # add basepair info to total data
        data.update(self._get_is_isolatable_info([motif_1, motif_2])) # add isolatable info to total data
        return data
        
    def _get_motif_info(self, motif_1, motif_2):
        motif_info_1 = parse_motif_indentifier(motif_1)
        motif_info_2 = parse_motif_indentifier(motif_2)
        return {
            "motif_1_id": motif_1,
            "motif_2_id": motif_2,
            "is_motif_1_unique": int(motif_1 in self.unique_motifs),
            "is_motif_2_unique": int(motif_2 in self.unique_motifs),
            "motif_1_type": motif_info_1[0],
            "motif_2_type": motif_info_2[0],
            "motif_1_size": motif_info_1[1],
            "motif_2_size": motif_info_2[1],
            "m_sequence_1": motif_info_1[2],
            "m_sequence_2": motif_info_2[2],
        }

    def _get_hbond_info(self, g):
        hbond_types = {
            "base-base": 0,
            "base-sugar": 0,
            "base-phos": 0,
            "sugar-sugar": 0,
            "phos-sugar": 0,
            "phos-phos": 0,
        }
        hbond_score = 0
        for _, row in g.iterrows():
            atom_types = sorted([row["atom_type_1"], row["atom_type_2"]])
            hbond_types[atom_types[0] + "-" + atom_types[1]] += 1
            hbond_score += row["score"]

        return {
            "num_base_base_hbonds": hbond_types["base-base"],
            "num_base_sugar_hbonds": hbond_types["base-sugar"],
            "num_base_phosphate_hbonds": hbond_types["base-phos"],
            "num_phosphate_sugar_hbonds": hbond_types["phos-sugar"],
            "num_phosphate_phosphate_hbonds": hbond_types["phos-phos"],
            "hbond_score": hbond_score,
            "num_hbonds": len(g),
        }

    def _get_residue_info(self, g):
        motif_1_res = []
        motif_2_res = []
        for _, row in g.iterrows():
            if row["res_1"] not in motif_1_res:
                motif_1_res.append(row["res_1"])
            if row["res_2"] not in motif_2_res:
                motif_2_res.append(row["res_2"])
        return {
            "motif_1_res": motif_1_res,
            "motif_2_res": motif_2_res,
            "num_res_1": len(motif_1_res),
            "num_res_2": len(motif_2_res),
        }

    def _get_basepair_info(self, motif_1, motif_2):
        df_basepairs = self.df_basepairs[
            (self.df_basepairs["motif_1"] == motif_1) & (self.df_basepairs["motif_2"] == motif_2)
        ]
        num_wc_pairs = 0
        wc_basepair_hbond_score = 0
        basepair_hbond_score = 0
        for _, row in df_basepairs.iterrows():
            basepair_hbond_score += row["hbond_score"]
            if row["lw"] != "cWW":
                continue
            res_1_id = parse_residue_identifier(row["res_1"])["res_id"]
            res_2_id = parse_residue_identifier(row["res_2"])["res_id"]
            bp_type = res_1_id + "-" + res_2_id
            if bp_type in wc_basepairs_w_gu:
                num_wc_pairs += 1
                wc_basepair_hbond_score += row["hbond_score"]
        return {
            "num_basepairs": len(df_basepairs),
            "basepair_hbond_score": basepair_hbond_score,
            "wc_basepair_hbond_score": wc_basepair_hbond_score,
            "num_wc_pairs": num_wc_pairs,
        }
    
    def _get_is_isolatable_info(self, motif_names):
        motifs = [self.motifs_by_name[m] for m in motif_names]
        df_basepairs = self.df_basepairs[
            (self.df_basepairs["motif_1"] == motif_names[0]) & (self.df_basepairs["motif_2"] == motif_names[1])
        ]
        res = []
        for m in motifs:
            for r in m.get_residues():
                if r.get_str() not in res:
                    res.append(r.get_str())
        total = 0
        base_hbond = 0
        ligand_hbond = 0
        for hb in motifs[0].hbonds + motifs[1].hbonds:
            # self hbond
            if hb.res_1.get_str() in res and hb.res_2.get_str() in res:
                continue
            if hb.res_type_2 == "LIGAND":
                continue
            if hb.res_type_2 == "SOLVENT":
                continue
            total += 1
            atom_type = get_nucleotide_atom_type(hb.atom_1)
            if atom_type == "BASE":
                base_hbond += 1
        num_external_basepairs = 0
        num_external_wc_basepairs = 0
        for _, row in df_basepairs.iterrows():
            if row["res_1"] not in res and row["res_2"] not in res:
                continue
            elif row["res_1"] in res and row["res_2"] in res:
                continue
            if row["lw"] == "cWW" and row["bp_type"] in wc_basepairs_w_gu:
                num_external_wc_basepairs += 1
            else:
                num_external_basepairs += 1
        return {
            "num_external_hbonds": total,
            "num_external_base_hbonds": base_hbond,
            "num_ligand_hbonds": ligand_hbond,
            "is_isolatable": int(total <= 5 and base_hbond <= 1 and ligand_hbond == 0),
            "num_external_basepairs": num_external_basepairs,
            "num_external_wc_basepairs": num_external_wc_basepairs,
        }
        

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
    path = os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json")
    if os.path.exists(path):
        return
    p = TertiaryContactProcessor()
    p.setup(pdb_id, unique_motifs)
    df = p.process()
    if len(df) == 0:
        return
    df.to_json(path, orient="records")


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


def find_tertiary_contact_basepairs(pdb_ids: List[str], processes: int = 1):
    """
    Find tertiary contact basepairs for multiple PDB structures using parallel processing.
    """
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "tc_basepairs"), exist_ok=True)
    # Process PDB IDs in parallel
    run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_id_for_tc_basepairs,
        processes=processes,
        batch_size=100,
        desc="Processing tertiary contact basepairs",
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


def generate_tertiary_contacts_release(processes: int = 1):
    path = os.path.join(DATA_PATH, "summaries", "tertiary_contacts")
    os.makedirs(path, exist_ok=True)
    json_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", "*.json")
    )
    results =[]
 
    df = pd.concat(results)
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
def run_find_tertiary_contact_basepairs(csv_path, processes):
    """ """
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].values
    find_tertiary_contact_basepairs(pdb_ids, processes)


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
    unique_motifs = list(df_unique_motifs["motif_name"].values)
    pdb_ids = df["pdb_id"].values
    find_tertiary_contacts(pdb_ids, unique_motifs, processes)


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def run_generate_tertiary_contacts_release(processes):
    """
    Process and filter tertiary contacts to get only unique motif interactions.
    """
    # finalized_tertiary_contacts("data/dataframes/tertiary_contacts/1GID.json")
    generate_tertiary_contacts_release(processes)


@cli.command()
def count_tertiary_contacts():
    df = concat_dataframes_from_files(
        glob.glob(os.path.join(DATA_PATH, "dataframes", "tertiary_contacts", "*.json"))
    )
    df = df[~((df["is_motif_1_unique"] == False) & (df["is_motif_2_unique"] == False))]
    df.to_json(
        os.path.join(
            DATA_PATH,
            "summaries",
            "tertiary_contacts",
            "tertiary_contacts_w_transient.json",
        ),
        orient="records",
    )
    exit()


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
