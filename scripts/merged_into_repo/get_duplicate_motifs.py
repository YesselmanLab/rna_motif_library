import pandas as pd
import os
import glob
import json
import pickle
from typing import List
from multiprocessing import Pool
import multiprocessing as mp

from rna_motif_library.motif import get_cached_motifs, Motif
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import (
    get_pdb_ids,
    NonRedundantSetParser,
    NRSEntry,
    add_motif_name_columns,
    parse_motif_name,
)
from rna_motif_library.tranforms import superimpose_structures, rmsd


def check_residue_numbers(motif1, motif2):
    """Compare residue numbers between two motifs.

    Args:
        motif1: First motif object
        motif2: Second motif object

    Returns:
        bool: True if motifs have same residue numbers, False otherwise
    """
    res1_nums = sorted([r.num for r in motif1.get_residues()])
    res2_nums = sorted([r.num for r in motif2.get_residues()])

    if len(res1_nums) != len(res2_nums):
        return False

    return res1_nums == res2_nums


def check_for_duplicates(motifs, other_motifs, repr_pdb, child_pdb):
    data = []
    used_motifs = []
    for i, om in enumerate(other_motifs):
        best_repr = None
        best_rmsd = 1000
        coords_1 = om.get_c1prime_coords()
        for j, m in enumerate(motifs):
            if m.name in used_motifs:
                continue
            if m.sequence != om.sequence:
                continue
            try:
                if len(coords_1) < 2:
                    continue
                coords_2 = m.get_c1prime_coords()
                if len(coords_2) != len(coords_1):
                    continue
                rotated_coords_2 = superimpose_structures(coords_2, coords_1)
                rmsd_val = rmsd(coords_1, rotated_coords_2)
                if rmsd_val < best_rmsd:
                    best_rmsd = rmsd_val
                    best_repr = m
            except:
                print("issues", m.name, om.name)
                continue
        is_duplicate = False
        if best_rmsd < 0.20 * len(coords_1):
            is_duplicate = True
        best_repr_name = best_repr.name if best_repr is not None else None
        data.append(
            {
                "motif": om.name,
                "repr_motif": best_repr_name,
                "rmsd": best_rmsd,
                "is_duplicate": is_duplicate,
                "repr_pdb": repr_pdb,
                "child_pdb": child_pdb,
                "from_repr": True,
            }
        )
        if is_duplicate:
            used_motifs.append(best_repr)

    df = pd.DataFrame(data)
    return df


def get_motifs(pdb_id: str):
    try:
        motifs = get_cached_motifs(pdb_id)
    except:
        print("missing motifs", pdb_id)
        return []
    return motifs


def get_entry_motifs(motifs: List[Motif], entry: NRSEntry):
    keep_motifs = []
    for m in motifs:
        keep = True
        for r in m.get_residues():
            if r.chain_id not in entry.chain_ids:
                keep = False
                break
        if keep:
            keep_motifs.append(m)
    return keep_motifs


def generate_repr_df(repr_motifs, pdb_id, from_repr=True):
    df_repr = []
    for m in repr_motifs:
        df_repr.append(
            {
                "motif": m.name,
                "repr_motif": None,
                "rmsd": None,
                "is_duplicate": False,
                "repr_pdb": pdb_id,
                "child_pdb": None,
                "from_repr": from_repr,
            }
        )
    return pd.DataFrame(df_repr)


def process_set(args):
    set_id, repr_entry, child_entries = args
    path = os.path.join(
        DATA_PATH,
        "dataframes",
        "non_redundant_sets",
        f"{set_id}.csv",
    )
    if os.path.exists(path):
        return None
    # if len(child_entries) == 0:
    #    return None
    all_repr_motifs = get_motifs(repr_entry.pdb_id)
    repr_motifs = get_entry_motifs(all_repr_motifs, repr_entry)
    dfs = []
    df_repr = generate_repr_df(repr_motifs, repr_entry.pdb_id, True)
    dfs.append(df_repr)
    print(set_id)
    # print(repr_entry.pdb_id, len(repr_motifs))
    all_missed_motifs = []
    for child_entry in child_entries:
        all_other_motifs = get_motifs(child_entry.pdb_id)
        other_motifs = get_entry_motifs(all_other_motifs, child_entry)
        df = check_for_duplicates(
            repr_motifs, other_motifs, repr_entry.pdb_id, child_entry.pdb_id
        )
        if len(df) == 0:
            continue
        df_sub = df.query("rmsd == 1000.0")
        missed_motifs_names = df_sub["motif"].values
        missed_motifs = []
        for m in other_motifs:
            if m.name in missed_motifs_names:
                missed_motifs.append(m)
        if len(missed_motifs) != 0:
            all_missed_motifs.append(missed_motifs)
        df_keep = df.query("rmsd < 1000").copy()
        dfs.append(df_keep)
    for i in range(len(all_missed_motifs)):
        if len(all_missed_motifs[i]) == 0:
            continue
        pdb_id = parse_motif_name(all_missed_motifs[i][0].name)[-1]
        for j in range(i + 1, len(all_missed_motifs)):
            if len(all_missed_motifs[j]) == 0:
                continue
            child_pdb_id = parse_motif_name(all_missed_motifs[j][0].name)[-1]
            df = check_for_duplicates(
                all_missed_motifs[i],
                all_missed_motifs[j],
                pdb_id,
                child_pdb_id,
            )
            duplicates = df.query("is_duplicate == True")
            dfs.append(duplicates)

            duplicate_names = duplicates["motif"].values
            all_missed_motifs[j] = [
                m for m in all_missed_motifs[j] if m.name not in duplicate_names
            ]
        dfs.append(generate_repr_df(all_missed_motifs[i], pdb_id, False))

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df.to_csv(
        path,
        index=False,
    )
    return path


def run_multiprocess(func, args_list, num_cores=16):
    """
    Run a function in parallel using multiprocessing.

    Args:
        func: Function to run in parallel
        args_list: List of arguments to pass to the function
        num_cores: Number of CPU cores to use (default 16)

    Returns:
        List of results from running the function
    """

    with Pool(num_cores) as pool:
        results = pool.map(func, args_list)
        return results


def assign_duplicates():
    parser = NonRedundantSetParser()
    sets = parser.parse(os.path.join(DATA_PATH, "csvs", "nrlist_3.369_3.5A.csv"))

    # Convert sets iterator to list of args
    set_args = [
        (set_id, repr_entry, child_entries)
        for set_id, repr_entry, child_entries in sets
    ]

    return run_multiprocess(process_set, set_args)


def organize_duplicate_motifs():
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets", "*.csv")
    )
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype={"repr_pdb": str, "child_pdb": str})
        except:
            continue
        df = df.query("is_duplicate == True").copy()
        dfs.append(df)
    df = pd.concat(dfs)
    df["repr_pdb"] = df["repr_pdb"].astype(str)
    df["child_pdb"] = df["child_pdb"].astype(str)
    for i, g in df.groupby("repr_pdb"):
        g.to_csv(
            os.path.join(DATA_PATH, "dataframes", "duplicate_motifs", f"{i}.csv"),
            index=False,
        )


def get_aligned_motifs():
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets", "*.csv")
    )
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype={"repr_pdb": str, "child_pdb": str})
        except:
            continue
        df = df.query("rmsd < 1000").copy()
        dfs.append(df)
    df = pd.concat(dfs)
    df["repr_pdb"] = df["repr_pdb"].astype(str)
    df["child_pdb"] = df["child_pdb"].astype(str)
    df.to_csv("aligned_motifs.csv", index=False)


def get_unique_motifs():
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets", "*.csv")
    )
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype={"repr_pdb": str, "child_pdb": str})
        except:
            continue
        df = df.query("is_duplicate == False").copy()
        dfs.append(df)
    df = pd.concat(dfs)
    for i, g in df.groupby("repr_pdb"):
        g.to_csv(
            os.path.join(DATA_PATH, "dataframes", "unique_motifs", f"{str(i)}.csv"),
            index=False,
        )
    df.to_csv("unique_motifs.csv", index=False)


def get_repr_motifs():
    parser = NonRedundantSetParser()
    sets = parser.parse(os.path.join(DATA_PATH, "csvs", "nrlist_3.369_3.5A.csv"))
    unique_motifs = pd.read_csv("unique_motifs.csv")["motif"].values
    missing_pdb_ids = []
    final_repr_motifs = []
    for set_id, repr_entry, child_entries in sets:
        all_repr_motifs = get_motifs(repr_entry.pdb_id)
        repr_motifs = get_entry_motifs(all_repr_motifs, repr_entry)
        if len(repr_motifs) == 0 and repr_entry.pdb_id not in missing_pdb_ids:
            missing_pdb_ids.append(repr_entry.pdb_id)
            continue
        print(repr_entry.pdb_id, len(final_repr_motifs))
        for m in repr_motifs:
            if m.name not in unique_motifs:
                final_repr_motifs.append(m.name)
    df = pd.DataFrame({"motif": final_repr_motifs})
    print(missing_pdb_ids)
    print(len(missing_pdb_ids))
    df.to_csv("repr_motifs.csv", index=False)


def summerize_sets():
    df = pd.read_csv("unique_motifs.csv")
    df.to_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv"), index=False
    )
    df = add_motif_name_columns(df, "motif")
    for i, g in df.groupby("mtype"):
        data = []
        for j, h in g.groupby("msequence"):
            data.append(
                {
                    "count": len(h),
                    "msequence": j,
                    "repr_motif": h.iloc[0]["motif"],
                }
            )
        df = pd.DataFrame(data)
        df.sort_values(by="count", ascending=False, inplace=True)
        df.to_csv(os.path.join(DATA_PATH, "summaries", f"{i}.csv"), index=False)


def get_unique_residues():
    df = pd.read_csv(os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv"))
    unique_motifs = df["motif"].values
    df = add_motif_name_columns(df, "motif")
    data = []
    res_mapping = []
    for pdb_id, g in df.groupby("pdb_id"):
        print(pdb_id)
        res_to_motif_id = {}
        motifs = get_cached_motifs(pdb_id)
        res = []
        for m in motifs:
            if m.name not in unique_motifs:
                continue
            for r in m.get_residues():
                if r.get_str() not in res:
                    res.append(r.get_str())
                if r.get_str() not in res_to_motif_id:
                    res_to_motif_id[r.get_str()] = m.name
                else:
                    existing_motif = res_to_motif_id[r.get_str()]
                    if existing_motif.startswith("HELIX"):
                        res_to_motif_id[r.get_str()] = m.name

        data.append({"pdb_id": pdb_id, "residues": res})
        res_mapping.append({"pdb_id": res_to_motif_id})
    df = pd.DataFrame(data)
    df.to_json(
        os.path.join(DATA_PATH, "summaries", "unique_residues.json"), orient="records"
    )
    df_res_mapping = pd.DataFrame(res_mapping)
    df_res_mapping.to_json(
        os.path.join(DATA_PATH, "summaries", "res_mapping.json"), orient="records"
    )


def process_pdb_id(pdb_id, mtype, unique_motifs):
    try:
        all_motifs = get_cached_motifs(pdb_id)
        motifs = [m for m in all_motifs if m.mtype == mtype]
        motifs = [m for m in motifs if m.name in unique_motifs]
        return (pdb_id, motifs)
    except:
        print("missing", pdb_id)
        return (pdb_id, [])


def process_pdb_id_chunk(args):
    pdb_id_chunk, mtype, unique_motifs = args
    return [process_pdb_id(pdb_id, mtype, unique_motifs) for pdb_id in pdb_id_chunk]


def split_pdb_ids(pdb_ids, split_nums=16):
    chunk_size = len(pdb_ids) // split_nums
    pdb_id_chunks = [
        pdb_ids[i : i + chunk_size] for i in range(0, len(pdb_ids), chunk_size)
    ]
    while len(pdb_id_chunks) > split_nums:
        pdb_id_chunks[-2].extend(pdb_id_chunks[-1])
        pdb_id_chunks.pop()
    return pdb_id_chunks


def generate_pickle_files():
    pdb_ids = get_pdb_ids()
    pdb_id_chunks = split_pdb_ids(pdb_ids, 16)
    mtype = "TWOWAY"
    unique_motifs = pd.read_csv("unique_motifs.csv")["motif"].values
    # Create list of arguments for each chunk
    args = [(chunk, mtype, unique_motifs) for chunk in pdb_id_chunks]
    # Process chunks in parallel
    results = run_multiprocess(process_pdb_id_chunk, args)
    motifs = {}
    for result in results:
        for pdb_id, specific_motifs in result:
            print(pdb_id, len(specific_motifs))
            motifs[pdb_id] = specific_motifs
    pickle.dump(motifs, open("twoways.pkl", "wb"))


def main():
    parser = NonRedundantSetParser()
    sets = parser.parse(os.path.join(DATA_PATH, "csvs", "nrlist_3.369_3.5A.csv"))
    for set_id, repr_entry, child_entries in sets:
        process_set((set_id, repr_entry, child_entries))
    exit()
    generate_pickle_files()
    exit()
    get_unique_residues()
    df_res_mapping = pd.read_json(
        os.path.join(DATA_PATH, "summaries", "res_mapping.json")
    )
    final_d = {}
    for i, row in df_res_mapping.iterrows():
        first_m = list(row["pdb_id"].values())[0]
        pdb_id = first_m.split("-")[-2]
        final_d[pdb_id] = row["pdb_id"]
    json.dump(final_d, open("res_mapping.json", "w"))


if __name__ == "__main__":
    main()
