import pandas as pd


from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.chain import get_cached_protein_chains, write_chain_to_cif
from rna_motif_library.util import parse_residue_identifier


def main():
    df = pd.read_json("motif_protein_interactions.json")
    df.sort_values("num_hbonds", ascending=False, inplace=True)
    df = df.query("num_protein_chains == 1")
    row = df.iloc[1000]
    print(row)
    motifs = get_cached_motifs(row["pdb_id"])
    protein_chains = get_cached_protein_chains(row["pdb_id"])
    protein_chains_by_id = {}
    motif_by_name = {m.name: m for m in motifs}
    motif = motif_by_name[row["motif_id"]]
    motif.to_cif()
    protein_chain_ids = []
    for hbond in row["hbonds"]:
        res_info = parse_residue_identifier(hbond[2])
        if res_info["chain_id"] not in protein_chain_ids:
            protein_chain_ids.append(res_info["chain_id"])
    print(protein_chain_ids)
    for chain_id in protein_chain_ids:
        count = 0
        for c in protein_chains:
            if c[0].chain_id == chain_id:
                write_chain_to_cif(c, f"protein_chain_{chain_id}_{count}.cif")
                count += 1


if __name__ == "__main__":
    main()
