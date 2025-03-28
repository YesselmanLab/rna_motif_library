import os
import pandas as pd
import click

from rna_motif_library.motif import get_cached_motifs, get_cached_hbonds
from rna_motif_library.residue import are_residues_connected
from rna_motif_library.util import get_nucleotide_atom_type, get_pdb_ids
from rna_motif_library.settings import DATA_PATH


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
                "pdb_id": pdb_id,
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


@click.group()
def cli():
    pass


@cli.command()
def find_tc_hbonds():
    pdb_ids = get_pdb_ids()
    dfs = []
    for pdb_id in pdb_ids:
        if os.path.exists(
            os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv")
        ):
            continue
        try:
            df_tc = find_tertiary_interactions(pdb_id)
            df_tc.to_csv(
                os.path.join(DATA_PATH, "dataframes", "tc_hbonds", f"{pdb_id}.csv"),
                index=False,
            )
            dfs.append(df_tc)
        except Exception as e:
            print(pdb_id)
            continue
    # df = pd.concat(dfs)
    # df.to_csv("tertiary_contacts_hbonds.csv", index=False)


@cli.command()
def find_tertiary_contacts():
    all_motifs = {}
    os.makedirs("tcs", exist_ok=True)
    df = pd.read_csv("tertiary_contacts_hbonds.csv")
    count = 0
    data = []
    for i, g in df.groupby(["motif_1", "motif_2"]):
        if len(g) < 3:
            continue
        hbond_types = {
            "base-base": 0,
            "base-sugar": 0,
            "base-phos": 0,
            "sugar-sugar": 0,
            "phos-sugar": 0,
            "phos-phos": 0,
        }
        if g.iloc[0]["pdb_id"] not in all_motifs:
            try:
                motifs = get_cached_motifs(g.iloc[0]["pdb_id"])
                all_motifs[g.iloc[0]["pdb_id"]] = motifs
            except Exception as e:
                continue
        else:
            motifs = all_motifs[g.iloc[0]["pdb_id"]]
        motifs_by_name = {m.name: m for m in motifs}
        motif_1 = motifs_by_name[g.iloc[0]["motif_1"]]
        motif_2 = motifs_by_name[g.iloc[0]["motif_2"]]
        hbond_score = 0
        for j, row in g.iterrows():
            atom_types = sorted([row["atom_type_1"], row["atom_type_2"]])
            hbond_types[atom_types[0] + "-" + atom_types[1]] += 1
            hbond_score += row["score"]
        data.append(
            {
                "motif_1": motif_1.name,
                "motif_2": motif_2.name,
                "mtype_1": motif_1.mtype,
                "mtype_2": motif_2.mtype,
                "m_size_1": motif_1.size,
                "m_size_2": motif_2.size,
                "num_hbonds": len(g),
                "hbond_score": hbond_score,
                "base-base": hbond_types["base-base"],
                "base-sugar": hbond_types["base-sugar"],
                "base-phos": hbond_types["base-phos"],
                "phos-sugar": hbond_types["phos-sugar"],
                "phos-phos": hbond_types["phos-phos"],
                "pdb_id": g.iloc[0]["pdb_id"],
            }
        )
    df = pd.DataFrame(data)
    df.to_csv("tertiary_contacts_summary.csv", index=False)


if __name__ == "__main__":
    cli()
