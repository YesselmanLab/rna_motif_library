import pandas as pd

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.residue import get_cached_residues, residues_to_cif_file


def get_motif_residues(res_list, residues: dict):
    m_residues = []
    for res in res_list:
        if res in residues:
            m_residues.append(residues[res])
    return m_residues


def main():
    motifs = get_cached_motifs("4V9F")
    motifs_by_name = {m.name: m for m in motifs}
    residues = get_cached_residues("4V9F")
    # df = pd.read_json("4V9F_hairpins.json")
    df = pd.read_json("data/dataframes/atlas_motifs_compared/4V9F.json")
    df = df.query("mtype == 'HAIRPIN'")
    for i, row in df.iterrows():
        if row["in_our_db"] and row["in_other_db"]:
            m_residues = get_motif_residues(row["residues"], residues)
            residues_to_cif_file(m_residues, f"BOTH_{row['motif']}.cif")
        elif row["in_our_db"]:
            m_residues = get_motif_residues(row["residues"], residues)
            residues_to_cif_file(m_residues, f"RSIA_{row['motif']}.cif")
        elif row["in_other_db"]:
            m_residues = get_motif_residues(row["residues"], residues)
            residues_to_cif_file(m_residues, f"ATLAS_{row['motif']}.cif")


if __name__ == "__main__":
    main()
