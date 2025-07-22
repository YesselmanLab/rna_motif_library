import click
import os
import glob
import pandas as pd
from typing import List

from rna_motif_library.motif import get_cached_motifs, get_motifs_from_json, Motif
from rna_motif_library.motif_factory import (
    get_cww_basepairs,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    PDBStructureData,
    HelixFinder,
    MotifFactory,
    get_singlet_pairs,
    get_basepair_ends_for_strands,
)
from rna_motif_library.chain import get_cached_chains, Chains
from rna_motif_library.basepair import Basepair
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    concat_dataframes_from_files,
)


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
            # dont know if i need this
            end_bp_residues = {end_bp.res_2.get_str(), end_bp.res_1.get_str()}
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
            return 0
    return 1


def check_motifs_in_pdb(pdb_id: str):
    path = f"data/dataframes/check_motifs/{pdb_id}.csv"
    if os.path.exists(path):
        return
    motifs = get_cached_motifs(pdb_id)
    helices = [m for m in motifs if m.mtype == "HELIX"]
    non_helix_motifs = [m for m in motifs if m.mtype != "HELIX"]
    pdb_data = get_pdb_structure_data(pdb_id)
    cww_basepairs_lookup_min = get_cww_basepairs(
        pdb_data, min_two_hbond_score=0.0, min_three_hbond_score=0.0
    )
    cww_basepairs_lookup = get_cww_basepairs(pdb_data)
    singlet_pairs = get_singlet_pairs(cww_basepairs_lookup, pdb_data.chains)
    data = []
    for m in non_helix_motifs:
        if m.mtype == "UNKNOWN":
            continue
        flanking_helices = 1
        if not check_motif_is_flanked_by_helices(m, helices, pdb_data.chains):
            flanking_helices = 0
        contains_helix = 1
        pdb_data_for_residues = get_pdb_structure_data_for_residues(
            pdb_data, m.get_residues()
        )
        hf = HelixFinder(pdb_data_for_residues, cww_basepairs_lookup_min, [])
        m_helices = hf.get_helices()
        contains_helix = 0
        if len(m_helices) > 0:
            contains_helix = 1
        has_singlet_pair = 0
        has_singlet_pair_end = 0
        for bp in m.basepairs:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in singlet_pairs:
                has_singlet_pair += 1
        for bp in m.basepair_ends:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in singlet_pairs:
                has_singlet_pair_end += 1
        data.append(
            {
                "pdb_id": pdb_id,
                "motif_name": m.name,
                "motif_type": m.mtype,
                "flanking_helices": flanking_helices,
                "contains_helix": contains_helix,
                "has_singlet_pair": has_singlet_pair,
                "has_singlet_pair_end": has_singlet_pair_end,
            }
        )
    for m in helices:
        other_helices = [h for h in helices if h != m]
        flanking_helices = 0
        for h in other_helices:
            if do_motifs_share_end_basepairs(m, h):
                flanking_helices += 1
        data.append(
            {
                "pdb_id": pdb_id,
                "motif_name": m.name,
                "motif_type": m.mtype,
                "flanking_helices": 0,
                "contains_helix": 0,
                "has_singlet_pair": 0,
                "has_singlet_pair_end": 0,
            }
        )
    df = pd.DataFrame(data)
    df["flanking_helices"] = df["flanking_helices"].astype(int)
    df["contains_helix"] = df["contains_helix"].astype(int)
    df["has_singlet_pair"] = df["has_singlet_pair"].astype(int)
    df["has_singlet_pair_end"] = df["has_singlet_pair_end"].astype(int)
    df.to_csv(path, index=False)


@click.group()
def cli():
    pass


def process_pdb_for_large_motifs(pdb_id: str) -> List[dict]:
    """Process a single PDB to find large motifs.

    Args:
        pdb_id (str): The PDB ID to process

    Returns:
        List[dict]: List of dictionaries containing information about large motifs found
    """
    data = []
    try:
        motifs = get_cached_motifs(pdb_id)
    except:
        return data

    path = os.path.join("data", "summaries", "non_redundant_motifs.csv")
    unique_motifs = list(pd.read_csv(path)["motif_name"].values)

    for motif in motifs:
        if motif.name not in unique_motifs:
            continue
        if motif.mtype == "UNKNOWN":
            continue
        if motif.mtype == "NWAY":
            continue
        if len(motif.get_residues()) > 50:
            data.append(
                {
                    "pdb_id": pdb_id,
                    "motif_name": motif.name,
                    "motif_type": motif.mtype,
                    "num_residues": len(motif.get_residues()),
                }
            )

    return data


@cli.command()
def find_large_motifs():
    pdb_ids = get_pdbs_ids_from_jsons("motifs")

    # Process PDBs in parallel batches
    all_data = run_w_processes_in_batches(
        pdb_ids, process_pdb_for_large_motifs, 20, 200
    )

    # Flatten the list of lists into a single list
    flat_data = [item for sublist in all_data for item in sublist]

    # Save to CSV
    df = pd.DataFrame(flat_data)
    df.to_csv("large_motifs.csv", index=False)

    print(f"Total large motifs found: {len(flat_data)}")


def extend_motif_with_basepairs(motif: Motif, pdb_data: PDBStructureData):
    ss_strand = motif.strands[0]
    prev_res = pdb_data.chains.get_previous_residue_in_chain(ss_strand[0])
    next_res = pdb_data.chains.get_next_residue_in_chain(ss_strand[-1])
    if prev_res is not None:
        ss_strand = [prev_res] + ss_strand
    if next_res is not None:
        ss_strand = ss_strand + [next_res]
    motif.strands = [ss_strand]
    return motif


def process_pdb_for_sstrand_overlap(pdb_id: str) -> dict:
    """Process a single PDB to check SSTRAND end overlaps.

    Args:
        pdb_id (str): The PDB ID to process

    Returns:
        dict: Dictionary containing information about SSTRAND overlaps
    """
    try:
        motifs = get_cached_motifs(pdb_id)
        pdb_data = get_pdb_structure_data(pdb_id)
        sstrands = [m for m in motifs if m.mtype == "SSTRAND"]
        for m in sstrands:
            extend_motif_with_basepairs(m, pdb_data)
        strands = [m.strands[0] for m in sstrands]
        mf = MotifFactory(pdb_data)
        cww_basepairs = mf.cww_basepairs_lookup
        non_canonical_motifs = mf.get_non_canonical_motifs(
            strands, cww_basepairs.values()
        )
        # Find motifs that are in sstrands but not in non_canonical_motifs
        for m in sstrands:
            m.to_cif()
        for m in non_canonical_motifs:
            if len(m.strands) > 1:
                m.to_cif()
                for strand in m.strands:
                    for res in strand:
                        print(res.get_str(), end=" ")
                    print()
        exit()

        if len(non_canonical_motifs) != len(sstrands):
            return {
                "pdb_id": pdb_id,
                "num_sstrands": len(sstrands),
                "num_non_canonical": len(non_canonical_motifs),
                "has_overlap": len(non_canonical_motifs) > 0,
            }
        return None
    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")
        return None


@cli.command()
def check_motifs():
    os.makedirs("data/dataframes/check_motifs", exist_ok=True)
    pdb_ids = get_pdbs_ids_from_jsons("motifs")
    run_w_processes_in_batches(pdb_ids, check_motifs_in_pdb, 10, 100)


@cli.command()
def check_motifs_analysis():
    file_paths = glob.glob("data/dataframes/check_motifs/*.csv")
    df = concat_dataframes_from_files(file_paths)
    df.to_csv("check_motifs_analysis.csv", index=False)


@cli.command()
def check_motif():
    motifs = get_cached_motifs("7OTC")
    # motifs = get_motifs_from_json("5UQ7_motifs.json")
    for m in motifs:
        res_strs = [r.get_str() for r in m.get_residues()]
        if "a-U-1030-" in res_strs:
            print(m.name)
            m.to_cif()
            exit()


@cli.command()
def check_sstrand_end_overlap():
    """Check SSTRAND end overlaps across all PDBs in parallel."""
    pdb_ids = get_pdbs_ids_from_jsons("motifs")
    process_pdb_for_sstrand_overlap("7OTC")
    exit()
    # Process PDBs in parallel batches
    all_data = run_w_processes_in_batches(
        pdb_ids, process_pdb_for_sstrand_overlap, 10, 100
    )
    # Convert results to DataFrame, filtering out None values
    df = pd.DataFrame([result for result in all_data if result is not None])
    # Save results
    df.to_csv("sstrand_overlap_analysis.csv", index=False)


if __name__ == "__main__":
    cli()
