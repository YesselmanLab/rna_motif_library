import pandas as pd
import os

from rna_motif_library.motif import ResidueId, get_cached_motifs
from rna_motif_library.chain import get_rna_chains

def get_residue_ids_from_hyperlink(hyperlink):
    start_idx = hyperlink.find('"') + 1
    end_idx = hyperlink.find('"', start_idx)
    url = hyperlink[start_idx:end_idx]

    # Get the unitid part which contains the residue IDs
    unitid_part = url.split("/")[-1]

    # Split into individual residue IDs
    residue_ids = unitid_part.split(",")
    return residue_ids


def get_lora_data():
    df = pd.read_csv("LORA.tsv", sep="\t")
    data = []
    for i, row in df.iterrows():
        hyperlink = row["RES"]
        residue_ids = get_residue_ids_from_hyperlink(hyperlink)
        # Parse each residue ID into standard format
        parsed_residues = []
        for res_id in residue_ids:
            res = ResidueId.from_string(res_id)
            parsed_residues.append(res.get_str())
        print(parsed_residues)
        data.append(
            {
                "pdb_id": residue_ids[0].split("|")[0],
                "residues": parsed_residues,
                "description": row["DESCRIPTION"],
                "secondary_structure": row["SSE"],
            }
        )
    df = pd.DataFrame(data)
    df.to_json("lora_data.json", orient="records")


def get_res_to_motif_mapping(motifs):
    res_to_motif_id = {}
    for m in motifs:
        for r in m.get_residues():
            if r.get_str() not in res_to_motif_id:
                res_to_motif_id[r.get_str()] = m.name
            else:
                existing_motif = res_to_motif_id[r.get_str()]
                if existing_motif.startswith("HELIX"):
                    res_to_motif_id[r.get_str()] = m.name
    return res_to_motif_id


def main():
    df = pd.read_json("lora_data.json")
    df["connected_strands"] = False
    os.makedirs("lora_contacts", exist_ok=True)
    count = 0
    for pdb_id, g in df.groupby("pdb_id"):
        try:
            motifs = get_cached_motifs(pdb_id)
        except Exception as e:
            print(f"Error getting motifs for {pdb_id}: {e}")
            continue
        motif_by_name = {m.name: m for m in motifs}
        res_to_motif_id = get_res_to_motif_mapping(motifs)
        for i, row in g.iterrows():
            residues = row["residues"]
            motif_ids = []
            for r in residues:
                if r in res_to_motif_id:
                    motif_ids.append(res_to_motif_id[r])
            motif_ids = list(set(motif_ids))
            motifs = [motif_by_name[m] for m in motif_ids]
            strands = []
            residues = []
            for m in motifs:
                strands.extend(m.strands)
                res = m.get_residues()
                for r in res:
                    if r not in residues:
                        residues.append(r)
            new_strands = get_rna_chains(residues)
            if len(new_strands) < len(strands):
                df.loc[i, "connected_strands"] = True
                continue

            print(count)
            print(motif_ids)
            print(len(strands), len(new_strands))
            os.makedirs(f"lora_contacts/{count}", exist_ok=True)
            try:
                for m in motifs:
                    m.to_cif(f"lora_contacts/{count}/{m.name}.cif")
            except Exception as e:
                print(f"Error writing motifs for {pdb_id}: {e}")
            count += 1



if __name__ == "__main__":
    main()
