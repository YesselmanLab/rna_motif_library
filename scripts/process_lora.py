import pandas as pd
import os
import click

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
    df = pd.read_json("unique_tertiary_contacts.json")
    df_loras = pd.read_json("lora_data.json")
    print(len(df_loras))
    df_loras["connected_strands"] = False
    df_loras["in_our_db"] = False
    df_loras["best_score"] = 0
    df_loras["hbond_score"] = 0
    df_loras["more_than_2_motifs"] = 0
    os.makedirs("lora_contacts", exist_ok=True)
    count = 0
    for pdb_id, g in df_loras.groupby("pdb_id"):
        try:
            motifs = get_cached_motifs(pdb_id)
        except Exception as e:
            print(f"Error getting motifs for {pdb_id}: {e}")
            continue
        try:
            df_hbonds = pd.read_csv(f"data/dataframes/tc_hbonds/{pdb_id}.csv")
        except Exception as e:
            df_hbonds = pd.DataFrame()
        motif_by_name = {m.name: m for m in motifs}
        res_to_motif_id = get_res_to_motif_mapping(motifs)
        for i, row in g.iterrows():
            residues = row["residues"]
            motif_ids = []
            seen_residues = []
            for r in residues:
                if r in res_to_motif_id:
                    seen_residues.append(r)
                    motif_ids.append(res_to_motif_id[r])
            if len(seen_residues) != len(residues):
                print(f"Error: {pdb_id} len(residues): {len(residues)} len(seen_residues): {len(seen_residues)}")
                continue
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
            hbond_score = 0
            for j, row in df_hbonds.iterrows():
                if row["motif_1"] in motif_ids or row["motif_2"] in motif_ids:
                    hbond_score += row["score"]
            df_loras.loc[i, "hbond_score"] = hbond_score
            if len(motif_ids) > 2:
                df_loras.loc[i, "more_than_2_motifs"] = True
                continue
            if len(new_strands) < len(strands):
                df_loras.loc[i, "connected_strands"] = True
                if count > 10:
                    for m in motifs:
                        m.to_cif(f"lora_{m.name}.cif")
                    exit()
                continue
            res = [r.get_str() for r in residues]
            best_score = 0
            best_index = None
            df_loras.loc[i, "hbond_score"] = hbond_score
            df_sub = df[df["pdb_id"] == pdb_id]
            for j, other_row in df_sub.iterrows():
                found_1 = 0
                found_2 = 0
                for r in other_row["motif_1_res"]:
                    if r in res:
                        found_1 += 1
                for r in other_row["motif_2_res"]:
                    if r in res:
                        found_2 += 1
                score = found_1 + found_2
                if found_1 == 0 or found_2 == 0:
                    score = 0
                score = score / (
                    len(other_row["motif_1_res"]) + len(other_row["motif_2_res"])
                )
                if score > best_score:
                    best_score = score
                    best_index = j
            if best_index is None:
                continue
            if best_score > 0.1:
                df_loras.loc[i, "in_our_db"] = True
            df_loras.loc[i, "best_score"] = best_score
            os.makedirs(f"lora_contacts/{count}", exist_ok=True)
            try:
                for m in motifs:
                    m.to_cif(f"lora_contacts/{count}/lora_{m.name}.cif")
            except Exception as e:
                print(f"Error writing motifs for {pdb_id}: {e}")
            """row = df_sub.loc[best_index]
            m1 = motif_by_name[row["motif_1"]]
            m2 = motif_by_name[row["motif_2"]]
            m1.to_cif(f"lora_contacts/{count}/{m1.name}.cif")
            m2.to_cif(f"lora_contacts/{count}/{m2.name}.cif")
            """
            print(count, best_score)
            count += 1
    df_loras.to_json("lora_contacts_processed.json", orient="records")


if __name__ == "__main__":
    main()
