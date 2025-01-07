import pandas as pd
import numpy as np
import glob
import math
import os


from rna_motif_library.classes import Residue, Basepair
from rna_motif_library.resources import ResidueManager, BasepairManager
from rna_motif_library.tranforms import (
    align_basepair_to_identity,
    angle_between_base_planes,
    NucleotideReferenceFrameGenerator,
)
from rna_motif_library.util import get_cif_header_str


def basepair_to_cif(res1: Residue, res2: Residue, path: str):
    f = open(path, "w")
    f.write(get_cif_header_str())
    acount = 1
    for res in [res1, res2]:
        res_str, acount = res.to_cif_str(acount)
        f.write(res_str)
    f.close()


def generate_exemplar_basepairs_old():
    rm = ResidueManager()
    bp_manager = BasepairManager()
    df = pd.read_csv("canonical_basepairs.csv")
    bp_ids = ["AU", "CG", "GU"]
    df_bp_hbonds = pd.read_csv("rna_motif_library/resources/basepair_hbonds.csv")
    df_bp_hbonds = df_bp_hbonds[df_bp_hbonds["basepair_type"] == "AU_cWW"]
    fg = NucleotideReferenceFrameGenerator()
    count = 0

    for bp_id in bp_ids:
        count = 0
        best_angle = 1000
        g = df[(df["bp_id"] == bp_id) & (df["bp_name"] == "cWW")]
        print("starting: ", len(g))
        for i, row in g.iterrows():
            bp = bp_manager.get_basepair(row["res_1"], row["res_2"], row["pdb_code"])
            res1 = rm.get_residue(bp.res_1.get_str(), row["pdb_code"])
            res2 = rm.get_residue(bp.res_2.get_str(), row["pdb_code"])
            if res1 is None or res2 is None:
                print(row["res_1"], row["res_2"], row["pdb_code"])
                continue
            frame1 = fg.get_reference_frame(res1)
            frame2 = fg.get_reference_frame(res2)
            angle = angle_between_base_planes(frame1, frame2)
            if angle < best_angle:
                print(angle)
                _, aligned_res1, aligned_res2 = align_basepair_to_identity(res1, res2)
                basepair_to_cif(
                    aligned_res1,
                    aligned_res2,
                    f"exemplars/{count}_{round(angle, 2)}.cif",
                )
                best_bp = bp
                best_angle = angle
            count += 1
            if count % 100 == 0:
                print(count)
        exit()


def generate_exemplar_basepairs():
    rm = ResidueManager()
    bp_manager = BasepairManager()
    df = pd.read_csv("canonical_basepairs.csv")
    bp_ids = ["AU", "CG", "GU"]
    df_bp_hbonds = pd.read_csv("rna_motif_library/resources/basepair_hbonds.csv")
    df_bp_hbonds = df_bp_hbonds[df_bp_hbonds["basepair_type"] == "AU_cWW"]
    fg = NucleotideReferenceFrameGenerator()
    count = 0

    for bp_id in bp_ids:
        count = 0
        best_angle = 1000
        g = df[(df["bp_id"] == bp_id) & (df["bp_name"] == "cWW")]
        print("starting: ", len(g))
        for i, row in g.iterrows():
            bp = bp_manager.get_basepair(row["res_1"], row["res_2"], row["pdb_code"])
            res1 = rm.get_residue(bp.res_1.get_str(), row["pdb_code"])
            res2 = rm.get_residue(bp.res_2.get_str(), row["pdb_code"])
            if res1 is None or res2 is None:
                print(row["res_1"], row["res_2"], row["pdb_code"])
                continue
            frame1 = fg.get_reference_frame(res1)
            frame2 = fg.get_reference_frame(res2)
            angle = angle_between_base_planes(frame1, frame2)
            if angle < best_angle:
                print(angle)
                _, aligned_res1, aligned_res2 = align_basepair_to_identity(res1, res2)
                basepair_to_cif(
                    aligned_res1,
                    aligned_res2,
                    f"exemplars/{count}_{round(angle, 2)}.cif",
                )
                best_bp = bp
                best_angle = angle
            count += 1
            if count % 100 == 0:
                print(count)
        exit()


def get_all_basepairs():
    json_files = glob.glob(os.path.join("data", "dataframes", "basepairs", "*.json"))
    dfs = []
    for file in json_files:
        pdb_code = file.split("/")[-1].split(".")[0]
        df = pd.read_json(file)
        df["pdb_code"] = pdb_code
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_pickle("all_basepairs.pkl")


def main():
    os.makedirs("exemplars", exist_ok=True)
    rm = ResidueManager()
    fg = NucleotideReferenceFrameGenerator()
    df = pd.read_pickle("all_basepairs.pkl")
    df_hbonds = pd.read_csv("rna_motif_library/resources/basepair_hbonds.csv")
    for (bp_type, lw), g in df.groupby(["bp_type", "lw"]):
        if not all(c in "ACGU" for c in bp_type):
            continue
        # if not (bp_type == "AA" and lw == "cHS"):
        #    continue
        key = f"{bp_type}_{lw}"
        if key not in df_hbonds["basepair_type"].values:
            continue
        g_sub = g.query(
            "abs(buckle) < 20 and abs(propeller) < 20 and abs(opening) < 20"
        )
        if len(g_sub) == 0:
            print(bp_type, lw, "no good basepairs found")
            g_sub = g
        g_sub = g_sub.sort_values(by="hbond_score", ascending=False)
        row = g_sub.iloc[0]
        print(bp_type, lw, len(g_sub), row["hbond_score"])
        res1 = rm.get_residue(row["res_1"], row["pdb_code"])
        res2 = rm.get_residue(row["res_2"], row["pdb_code"])
        frame1 = fg.get_reference_frame(res1)
        frame2 = fg.get_reference_frame(res2)
        angle = angle_between_base_planes(frame1, frame2)
        count = 0
        _, aligned_res1, aligned_res2 = align_basepair_to_identity(res1, res2)
        basepair_to_cif(
            aligned_res1,
            aligned_res2,
            f"exemplars/{key}.cif",
        )
    exit()
    df = df.sort_values(by="hbond_score", ascending=False)
    row = df.iloc[0]
    print(row)
    res1 = rm.get_residue(row["res_1"], row["pdb_code"])
    res2 = rm.get_residue(row["res_2"], row["pdb_code"])
    frame1 = fg.get_reference_frame(res1)
    frame2 = fg.get_reference_frame(res2)
    angle = angle_between_base_planes(frame1, frame2)
    count = 0
    _, aligned_res1, aligned_res2 = align_basepair_to_identity(res1, res2)
    basepair_to_cif(
        aligned_res1,
        aligned_res2,
        f"exemplars/{count}_{round(angle, 2)}.cif",
    )


if __name__ == "__main__":
    main()
