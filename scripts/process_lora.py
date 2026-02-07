import pandas as pd
import os
import itertools
from collections import defaultdict

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_analysis import ResidueId
from rna_motif_library.chain import get_rna_chains, write_chain_to_cif
from rna_motif_library.util import parse_motif_indentifier

DATA_PATH = "data/summaries/tertiary_contacts/lora"

# process the lora data from TableS8 as recommended by the authors #####################

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
    df.to_json(os.path.join(DATA_PATH, "lora_data.json"), orient="records")

# compare the lora data to our database ################################################

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
                    "motif_1_id": motif_names[0],
                    "motif_2_id": motif_names[1],
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
                "num_hbonds": row["num_hbonds"],
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


def generate_initial_comparision():
    df_loras = pd.read_json(os.path.join(DATA_PATH, "lora_data.json"))
    df_loras["index"] = list(range(len(df_loras)))
    df_loras["connected_strands"] = False
    df_loras["in_our_db"] = False
    df_loras["best_score"] = 0
    df_loras["hbond_score"] = 0
    df_loras["more_than_2_motifs"] = 0
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
        tcc = TertiaryContactComparer()
        df_compared = tcc.compare(pdb_id, df_loras, df_sub_tc)
        df_compareds.append(df_compared)
    df_compareds = pd.concat(df_compareds)
    df_compareds.to_json(os.path.join(DATA_PATH, "lora_compared.json"), orient="records") 

# get more information about tertiary not found in LORA ################################

def get_more_information_about_tertiary_not_found_in_lora():
    df_compared = pd.read_json(os.path.join(DATA_PATH, "lora_compared.json"))
    df_compared = df_compared[["motif_1_id", "motif_2_id", "in_our_db", "in_their_db"]]
    df_compared["pdb_id"] = df_compared["motif_1_id"].apply(lambda x: parse_motif_indentifier(x)[-1])
    dfs = []
    for pdb_id, g in df_compared.groupby("pdb_id"):
        df_sub_tc = pd.read_json(f"data/dataframes/tertiary_contacts/{pdb_id}.json")
        dfs.append(df_sub_tc)
    df_tcs = pd.concat(dfs)
    df_tcs.drop(columns=["pdb_id"], inplace=True)
    df_compared = pd.merge(df_compared, df_tcs, on=["motif_1_id", "motif_2_id"])
    df_compared.to_json(os.path.join(DATA_PATH, "lora_compared_w_features.json"), orient="records")


def main():
    # generate_initial_comparision()
    get_more_information_about_tertiary_not_found_in_lora()


if __name__ == "__main__":
    main()
