import click
import os
import pandas as pd

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.settings import DATA_PATH


def get_pdbs_ids_from_jsons(jsons_dir: str):
    json_path = os.path.join(DATA_PATH, "jsons", jsons_dir)
    pdb_ids = []
    for file in os.listdir(json_path):
        if file.endswith(".json"):
            pdb_ids.append(file.split(".")[0])
    return pdb_ids


def split_motif_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split motif names into component parts.
    Motif names are in format: mtype-[components]-pdb_code-count
    where [components] varies based on number of strands.
    E.g. TWOWAY-5-4-GAAAG-UAAC-1GID-1

    Args:
        df: DataFrame containing motif_1 and motif_2 columns

    Returns:
        DataFrame with additional columns for split components
    """
    # Split motif names into parts
    df_split1 = df["motif_1"].str.split("-", expand=True)
    df_split2 = df["motif_2"].str.split("-", expand=True)

    # First column is always mtype
    df["mtype_1"] = df_split1[0]
    df["mtype_2"] = df_split2[0]

    # Last column is count, second-to-last is pdb_id
    df["count_1"] = pd.to_numeric(df_split1.iloc[:, -1])
    df["count_2"] = pd.to_numeric(df_split2.iloc[:, -1])
    df["pdb_1"] = df_split1.iloc[:, -2]
    df["pdb_2"] = df_split2.iloc[:, -2]

    # Store remaining middle components as sequences
    df["sequences_1"] = df_split1.iloc[:, 1:-2].apply(
        lambda x: "-".join(x.dropna()), axis=1
    )
    df["sequences_2"] = df_split2.iloc[:, 1:-2].apply(
        lambda x: "-".join(x.dropna()), axis=1
    )

    return df


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
def check_hairpins():
    pdb_ids = get_pdbs_ids_from_jsons("motifs")
    for pdb_id in pdb_ids:
        try:
            motifs = get_cached_motifs(pdb_id)
        except:
            continue
        hairpins = []
        for motif in motifs:
            if motif.mtype == "HAIRPIN":
                hairpins.append(motif)
        for h in hairpins:
            if len(h.get_residues()) < 5:
                print("too small", pdb_id, h.name, len(h.get_residues()))
            if len(h.get_residues()) > 10:
                print("too large", pdb_id, h.name, len(h.get_residues()))


if __name__ == "__main__":
    cli()
