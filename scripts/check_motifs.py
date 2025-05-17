import click
import os
import pandas as pd
from typing import List

from rna_motif_library.motif import get_cached_motifs, get_motifs_from_json, Motif
from rna_motif_library.motif_factory import (
    get_cww_basepairs,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    PDBStructureData,
    HelixFinder,
)
from rna_motif_library.chain import get_cached_chains, Chains
from rna_motif_library.basepair import Basepair
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import parse_motif_name


def get_pdbs_ids_from_jsons(jsons_dir: str):
    json_path = os.path.join(DATA_PATH, "jsons", jsons_dir)
    pdb_ids = []
    for file in os.listdir(json_path):
        if file.endswith(".json"):
            pdb_ids.append(file.split(".")[0])
    return pdb_ids


def do_motifs_share_end_basepairs(motif_1: Motif, motif_2: Motif) -> bool:
    """
    Check if two motifs share any end basepairs.

    Args:
        other (Motif): The other motif to check against

    Returns:
        bool: True if the motifs share any end basepairs, False otherwise
    """
    # Get all residue strings from end basepairs of both motifs
    self_end_residues = set()
    for bp in motif_1.basepair_ends:
        self_end_residues.add(bp.res_1.get_str())
        self_end_residues.add(bp.res_2.get_str())

    other_end_residues = set()
    for bp in motif_2.basepair_ends:
        other_end_residues.add(bp.res_1.get_str())
        other_end_residues.add(bp.res_2.get_str())

    # Check if there's any overlap in the end residues
    return bool(self_end_residues & other_end_residues)


def find_motifs_sharing_basepair(
    basepair: Basepair, motifs: List["Motif"]
) -> List["Motif"]:
    """
    Find all motifs that share the given end basepair.

    Args:
        basepair (Basepair): The basepair to search for
        motifs (List[Motif]): List of motifs to search through

    Returns:
        List[Motif]: List of motifs that share the given basepair
    """
    sharing_motifs = []
    basepair_residues = {basepair.res_1.get_str(), basepair.res_2.get_str()}
    for motif in motifs:
        for end_bp in motif.basepair_ends:
            end_bp_residues = {end_bp.res_1.get_str(), end_bp.res_2.get_str()}
            if basepair_residues == end_bp_residues:
                sharing_motifs.append(motif)
                break

    return sharing_motifs


def check_motif_is_flanked_by_helices(
    motif: Motif, helices: List[Motif], chains: Chains
) -> bool:
    for bp in motif.basepair_ends:
        res_1, res_2 = chains.get_residues_in_basepair(bp)
        if chains.is_chain_end(res_1) and chains.is_chain_end(res_2):
            continue
        shared_helices = find_motifs_sharing_basepair(bp, helices)
        if len(shared_helices) == 0:
            print(
                f"motif {motif.name} is not flanked by helices {bp.res_1.get_str()}-{bp.res_2.get_str()}"
            )
            return False
    return True


def is_next_bp_in_cww_basepairs(
    bp: Basepair, pdb_data: PDBStructureData, cww_basepairs_lookup: dict
) -> bool:
    res_1 = pdb_data.chains.get_residue_by_str(bp.res_1.get_str())
    res_2 = pdb_data.chains.get_residue_by_str(bp.res_2.get_str())
    chain_1 = pdb_data.chains.get_chain_for_residue(res_1)
    chain_2 = pdb_data.chains.get_chain_for_residue(res_2)
    if chain_1 is None or chain_2 is None:
        raise Exception(
            f"chain_1 or chain_2 is None for {bp.res_1.get_str()}-{bp.res_2.get_str()}"
        )
        return False
    if res_1 == chain_1[0]:
        next_res_1 = chain_1[1]
    else:
        next_res_1 = chain_1[-1]
    if res_2 == chain_2[0]:
        next_res_2 = chain_2[1]
    else:
        next_res_2 = chain_2[-1]
    key = f"{next_res_1.get_str()}-{next_res_2.get_str()}"
    if key in cww_basepairs_lookup:
        return True
    return False


@click.group()
def cli():
    pass


@cli.command()
def check_singlets():
    pdb_ids = get_pdbs_ids_from_jsons("motifs")
    for pdb_id in pdb_ids:
        try:
            motifs = get_cached_motifs(pdb_id)
        except:
            continue
        # Track residues and their motifs
        residue_to_motifs = {}
        # Build mapping of residues to motifs they appear in
        for motif in motifs:
            for residue in motif.get_residues():
                res_str = residue.get_str()
                if res_str not in residue_to_motifs:
                    residue_to_motifs[res_str] = []
                residue_to_motifs[res_str].append(motif.name)
        # Find residues that appear in multiple motifs
        shared_data = []
        for res_str, motif_names in residue_to_motifs.items():
            if len(motif_names) == 1:
                continue
            # Add each pair of motifs sharing this residue
            for i in range(len(motif_names)):
                for j in range(i + 1, len(motif_names)):
                    shared_data.append(
                        {
                            "pdb_id": pdb_id,
                            "residue": res_str,
                            "motif_1": motif_names[i],
                            "motif_2": motif_names[j],
                        }
                    )
        if len(shared_data) == 0:
            continue
        df = pd.DataFrame(shared_data)
        df = split_motif_names(df)
        for i, g in df.groupby(["mtype_1", "mtype_2"]):
            if i[0] != "HELIX" and i[1] != "HELIX":
                print(pdb_id, i)


@cli.command()
def find_large_motifs():
    # Create directories for each motif type
    motif_types = ["HAIRPIN", "HELIX", "NWAY", "SSTRAND", "TWOWAY"]
    for mtype in motif_types:
        os.makedirs(f"large_motifs/{mtype.lower()}", exist_ok=True)

    unique_motifs = list(pd.read_csv("unique_motifs.csv")["motif"].values)
    pdb_ids = get_pdbs_ids_from_jsons("motifs")
    count = 0

    data = []
    for pdb_id in pdb_ids:
        try:
            motifs = get_cached_motifs(pdb_id)
        except:
            continue

        for motif in motifs:
            if motif.name not in unique_motifs:
                continue
            if motif.mtype == "UNKNOWN":
                continue

            if len(motif.get_residues()) > 30:
                count += 1
                print(
                    "large",
                    count,
                    pdb_id,
                    motif.name,
                    motif.mtype,
                    len(motif.get_residues()),
                )
                output_dir = f"large_motifs/{motif.mtype.lower()}"
                try:
                    motif.to_cif(os.path.join(output_dir, motif.name + ".cif"))
                except:
                    print("error", motif.name)

                data.append(
                    {
                        "pdb_id": pdb_id,
                        "motif_name": motif.name,
                        "motif_type": motif.mtype,
                        "num_residues": len(motif.get_residues()),
                    }
                )

    print(f"Total large motifs found: {count}")

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("large_motifs.csv", index=False)


@cli.command()
def check_hairpins():
    # motifs = get_motifs_from_json("data/jsons/old_motifs/7R6Q.json")
    motifs = get_motifs_from_json("7R6Q_motifs.json")
    # motifs = get_cached_motifs("7UO0")
    helices = [m for m in motifs if m.mtype == "HELIX"]
    non_helix_motifs = [m for m in motifs if m.mtype != "HELIX"]
    pdb_data = get_pdb_structure_data("7R6Q")
    cww_basepairs_lookup = get_cww_basepairs(
        pdb_data, min_two_hbond_score=0.5, min_three_hbond_score=0.5
    )
    cww_basepairs_lookup_min = get_cww_basepairs(
        pdb_data, min_two_hbond_score=0.0, min_three_hbond_score=0.0
    )
    for m in non_helix_motifs:
        if not check_motif_is_flanked_by_helices(m, helices, pdb_data.chains):
            print("not flanked by helices", m.name)
        for bp in m.basepair_ends:
            if is_next_bp_in_cww_basepairs(bp, pdb_data, cww_basepairs_lookup):
                print(
                    "in cww_basepairs_lookup",
                    m.name,
                    bp.res_1.get_str(),
                    bp.res_2.get_str(),
                )
        pdb_data_for_residues = get_pdb_structure_data_for_residues(
            pdb_data, m.get_residues()
        )
        hf = HelixFinder(pdb_data_for_residues, cww_basepairs_lookup_min, [])
        m_helices = hf.get_helices()
        if len(m_helices) > 1:
            print("has helices", m.name)


@cli.command()
def check_motif():
    motifs = get_cached_motifs("6LKQ")
    for m in motifs:
        if m.mtype != "TWOWAY":
            continue
        if len(m.get_residues()) > 50:
            print(m.name, len(m.get_residues()))
            m.to_cif(m.name + ".cif")


if __name__ == "__main__":
    cli()
