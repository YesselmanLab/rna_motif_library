import pandas as pd

from rna_motif_library.motif import get_motifs_from_json

def main():
    PATH = "scripts/resources/motifs/"
    motifs = get_motifs_from_json(PATH + "/1GID.json")
    data = []
    for m in motifs:
        data.append({
            "pdb_id": "1GID",
            "mtype": m.mtype,
            "sequence": m.sequence,
            "residues": [r.get_str() for r in m.get_residues()], 
        })
    df = pd.DataFrame(data)
    df.to_json("test/resources/motifs/1GID.json", orient="records")

if __name__ == "__main__":
    main()
