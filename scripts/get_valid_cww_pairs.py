import pandas as pd 
import glob

from rna_motif_library.parallel_utils import concat_dataframes_from_files

def get_cww_basepairs():
    json_files = glob.glob("data/dataframes/basepairs/*.json")
    df = concat_dataframes_from_files(json_files)
    df = df.query("lw == 'cWW'")
    df["res_type_1"] = df["res_1"].apply(lambda x: x.split("-")[1])
    df["res_type_2"] = df["res_2"].apply(lambda x: x.split("-")[1])
    df["bp_type"] = df.apply(lambda x: get_bp_type(x["res_type_1"], x["res_type_2"]), axis=1)
    # these are real cWW basepairs dont need to look at them now
    exclude = ["C-G", "A-U", "G-U", "A-T", "C-I"]
    df = df[~df["bp_type"].isin(exclude)]
    # these are definitely not valid cWW basepairs
    exclude = ["A-G", "U-U", "C-U", "A-C", "A-A", "C-C", "G-G"]
    df = df[~df["bp_type"].isin(exclude)]
    df.to_json("cww_basepairs.json", orient="records")


def get_bp_type(res_1_type: str, res_2_type: str) -> str:
    e = [res_1_type, res_2_type]
    if e[0] > e[1]:
        return e[1] + "-" + e[0]
    else:
        return e[0] + "-" + e[1]


def find_valid_cww_pairs():
    df = pd.read_json("cww_basepairs.json")
    df_res_mapping = pd.read_csv("res_mapping.csv")
    df_ligand = pd.read_json("rna_ligand_interactions.json")
    df_ligand["ligand_id"] = df_ligand["ligand_res"].apply(lambda x: x.split("-")[1])
    seen = set(df_res_mapping["res_type"].values)
    ligs = set(df_ligand["ligand_id"].values)
    exclude = ["F7O", "75B", "05H", "RVP", "05K", "XMP", "QSK", "PPU", "P5P", "T2T", "OBX", "CVC", "M5M"]
    df = df[~df["res_type_1"].isin(exclude) & ~df["res_type_2"].isin(exclude)]
    df = df[~(df["res_type_1"].isin(seen)) & ~(df["res_type_2"].isin(seen))]
    df = df[~df["res_type_1"].isin(ligs) & (~df["res_type_2"].isin(ligs))]
    # there are valid
    exclude = ["I-U"]
    df = df[~df["bp_type"].isin(exclude)]
    # these are not valid cWW basepairs
    exclude = ["Y5P-Y5P", "BRU-C", "P5P-Y5P", "N-N", "A-B8T", "P5P-U", "A-Y5P", "6MZ-U", "A-I", "A-AG9", "G-NF2", "C-SDG", "C-P5P", "A-RSP", "G-Y5P", "C-MUM", "G-I"]
    df = df[~df["bp_type"].isin(exclude)]
    unsure = ["G-LV2", "A-M3X"]
    df = df[~df["bp_type"].isin(unsure)]
    grouped = df.groupby("bp_type").agg({
        "bp_type": "count",
        "hbond_score": "mean"
    }).rename(columns={
        "bp_type": "count",
        "hbond_score": "avg_hbond_score"
    })
    grouped = grouped[grouped["avg_hbond_score"] > 0.01]
    print(len(grouped))
    print(grouped.sort_values("count", ascending=False).head(10))

def main():
    # valid cWW basepairs
    valid_pairs = ["C-G", "A-U", "G-U", "A-T", "C-I", "I-U"]
    # random pairs that dont look right for manual inspection
    invalid_pairs = ["BRU-C", "P5P-Y5P", "N-N", "A-B8T", "P5P-U", "A-Y5P", "6MZ-U", "A-I", "A-AG9", "G-NF2", "C-SDG", "C-P5P", "A-RSP", "G-Y5P", "G-LV2", "A-M3X"]
    # residues that cant make a cWW pair or are ligands
    exclude_res = ["F7O", "ODC", "75B", "05H", "RVP", "05K", "XMP", "QSK", "PPU", "P5P", "T2T", "OBX", "CVC", "M5M", "Q1V", "NF2", "MUM", "Y5P"]
    # ligands cant be in cWW pairs
    df_ligand = pd.read_json("rna_ligand_interactions.json")
    df_ligand["ligand_id"] = df_ligand["ligand_res"].apply(lambda x: x.split("-")[1])
    ligs = set(df_ligand["ligand_id"].values)
    # map noncanonical res to likely coanonical to check is pairs are valid
    df_res_mapping = pd.read_csv("res_mapping.csv")
    res_mapping = {}
    for i, row in df_res_mapping.iterrows():
        res_mapping[row["res_type"]] = row["map_res_type"]
    all_valid_pairs = valid_pairs[:]
    df = pd.read_json("cww_basepairs.json")
    count = 0
    for pair, g in df.groupby("bp_type"):
        hbond_score = g["hbond_score"].mean()
        if hbond_score < 0.01:
            continue
        if pair in invalid_pairs:
            continue
        res_1, res_2 = pair.split("-")
        if res_1 in exclude_res or res_2 in exclude_res:
            continue  
        if res_1 in res_mapping and res_2 in res_mapping:
            res_1_map = res_mapping[res_1]
            res_2_map = res_mapping[res_2]
            bp_type = get_bp_type(res_1_map, res_2_map)
            if bp_type in valid_pairs:
                all_valid_pairs.append(pair)
        elif res_1 in ligs or res_2 in ligs:
            continue
        else:
            print("not accounted for: pair", pair, "count", count, "hbond_score", hbond_score)
            count += 1
    df = pd.DataFrame(all_valid_pairs, columns=["bp_type"])
    df.to_csv("rna_motif_library/resources/valid_cww_pairs.txt", index=False, header=False)




    
    
        


if __name__ == '__main__':
    main()