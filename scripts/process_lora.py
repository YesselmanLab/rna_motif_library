import pandas as pd
import os
import click
import itertools
from collections import defaultdict

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_analysis import ResidueId
from rna_motif_library.chain import get_rna_chains, write_chain_to_cif


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
    #df = pd.read_excel("TableS6.xlsx", sheet_name="LORA_MATCHES")
    df = pd.read_csv("LORA.tsv", sep="\t")
    data = []
    pdb_ids = set()
    for i, row in df.iterrows():
        hyperlink = row["RES"]
        residue_ids = get_residue_ids_from_hyperlink(hyperlink)
        # Parse each residue ID into standard format
        parsed_residues = []
        for res_id in residue_ids:
            res = ResidueId.from_string(res_id)
            parsed_residues.append(res.get_str())
        print(parsed_residues)
        pdb_ids.add(residue_ids[0].split("|")[0])
        data.append(
            {
                "pdb_id": residue_ids[0].split("|")[0],
                "residues": parsed_residues,
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


class TertiaryContactComparer:
    def __init__(self):
        pass

    def compare(self, pdb_id, df_loras, df_tc):
        try:
            motifs = get_cached_motifs(pdb_id)
        except Exception as e:  
            print(f"Error getting motifs for {pdb_id}: {e}")
            return pd.DataFrame()
        motif_by_name = {m.name: m for m in motifs}
        df_lora_tcs = {}
        for i, row in df_loras.iterrows():
            df_lora_tcs[row["index"]] = row["residues"]
        res_to_motif_id = get_res_to_motif_mapping(motifs)
        df_hbonds = pd.read_csv(f"data/dataframes/tc_hbonds/{pdb_id}.csv")
        hbond_dict = defaultdict(list)
        for i, row in df_hbonds.iterrows():
            motif_names = sorted([row["motif_1"], row["motif_2"]])
            hbond_dict[motif_names[0] + "_" + motif_names[1]].append(row["score"])
        our_tcs = {}
        for i, row in df_tc.iterrows():
            motif_names = sorted([row["motif_1_id"], row["motif_2_id"]])
            our_tcs[motif_names[0] + "_" + motif_names[1]] = row
        data = []
        seen = {}
        for i, row in df_loras.iterrows():
            motif_ids = self._get_interacting_motifs(row, res_to_motif_id)
            for m1, m2 in itertools.combinations(motif_ids, 2):
                are_motifs_connected = self._are_motifs_connected(
                    [motif_by_name[m1], motif_by_name[m2]]
                )
                motif_names = sorted([m1, m2])
                key = motif_names[0] + "_" + motif_names[1]
                if key in hbond_dict:
                    hbond_score = sum(hbond_dict[key])
                    hbond_num = len(hbond_dict[key])
                else:
                    hbond_score = 0
                    hbond_num = 0
                if key in our_tcs:
                    in_our_db = 1
                    seen[key] = 1
                else:
                    in_our_db = 0
                if hbond_num == 0:
                    continue
                if in_our_db == 1:
                    are_motifs_connected = 0
                row_data = {
                    "lora_index": i,
                    "motif_1_id": m1,
                    "motif_2_id": m2,
                    "hbond_score": hbond_score,
                    "num_hbonds": hbond_num,
                    "are_connected": 1 if are_motifs_connected else 0,
                    "in_our_db": in_our_db,
                    "in_their_db": 1,
                    "found_after": 0
                }
                data.append(row_data)

        for key, row in our_tcs.items():
            if key in seen:
                continue
            motif_1 = motif_by_name[row["motif_1_id"]]
            motif_2 = motif_by_name[row["motif_2_id"]] 
            residues_1 = []
            residues_2 = []
            found_1 = 0
            found_2 = 0
            for r in motif_1.get_residues():
                residues_1.append(r.get_str())
            for r in motif_2.get_residues():
                residues_2.append(r.get_str())
            for key, value in df_lora_tcs.items():
                for r in value:
                    if r in residues_1:
                        found_1 += 1
                    if r in residues_2:
                        found_2 += 1
            if found_1 > 0 and found_2 > 0:
                in_their_db = 1
                found_after = 1
                print("found after")
            else:
                found_after = 0
                in_their_db = 0
            row_data = {
                "lora_index": -1,
                "motif_1_id": row["motif_1_id"],
                "motif_2_id": row["motif_2_id"],
                "hbond_score": row["hbond_score"],
                "hbond_num": row["num_hbonds"],
                "are_connected": 0,
                "in_our_db": 1,
                "in_their_db": in_their_db,
                "found_after": found_after
            }
            data.append(row_data)
        return pd.DataFrame(data)

    def _get_interacting_motifs(self, row, res_to_motif_id):
        residues = row["residues"]
        motif_ids = set()
        for r in residues:
            if r in res_to_motif_id:
                motif_ids.add(res_to_motif_id[r])
        return motif_ids

    def _are_motifs_connected(self, motifs):
        strands = []
        residues = []
        motifs[0].to_cif("motif_1.cif")
        motifs[1].to_cif("motif_2.cif")
        for m in motifs:
            strands.extend(m.strands)
            res = m.get_residues()
            for r in res:
                if r not in residues:
                    residues.append(r)
        new_strands = get_rna_chains(residues)
        return len(new_strands) < len(strands)


def compare_lora_to_tertiary_contacts_for_pdb(args):
    pdb_id, df_loras, df_tc = args
    tcc = TertiaryContactComparer()
    df_compared = tcc.compare(pdb_id, df_loras, df_tc)
    return df_compared


def main():
    df_loras = pd.read_json("lora_data.json")
    df_loras["index"] = list(range(len(df_loras)))
    df_loras["connected_strands"] = False
    df_loras["in_our_db"] = False
    df_loras["best_score"] = 0
    df_loras["hbond_score"] = 0
    df_loras["more_than_2_motifs"] = 0
    os.makedirs("lora_contacts", exist_ok=True)
    count = 0
    pdb_ids = df_loras["pdb_id"].unique()
    pdb_data = []
    for pdb_id in pdb_ids:
        df_sub = df_loras[df_loras["pdb_id"] == pdb_id]
        try:
            df_sub_tc = pd.read_json(f"data/dataframes/tertiary_contacts/{pdb_id}.json")
        except Exception as e:
            print(f"Error getting tertiary contacts for {pdb_id}: {e}")
            continue
        pdb_data.append([pdb_id, df_sub, df_sub_tc])
    df_compareds = []
    for pdb_id, df_sub, df_sub_tc in pdb_data:
        print(pdb_id)
        df_compared = compare_lora_to_tertiary_contacts_for_pdb(
            (pdb_id, df_sub, df_sub_tc)
        )
        df_compareds.append(df_compared)
    df_compareds = pd.concat(df_compareds)
    print(len(df_compareds))
    df_compareds.to_json("lora_compared.json", orient="records")
    exit()
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
                print(
                    f"Error: {pdb_id} len(residues): {len(residues)} len(seen_residues): {len(seen_residues)}"
                )
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
