import pandas as pd
import json
import os

from rna_motif_library.cli import get_pdb_ids
from rna_motif_library.classes import Basepair, Hbond
from rna_motif_library.motif import get_motifs_from_json
from rna_motif_library.settings import DATA_PATH


def get_hbonds_from_json(json_file):
    """Read tertiary contact hbonds from json file"""
    with open(json_file) as f:
        data = json.load(f)
    hbonds = []
    for entry in data:
        hbond = Hbond.from_dict(entry["hbond"])
        entry["hbond"] = hbond
        hbonds.append(entry)
    return hbonds


def get_basepairs_from_json(json_file):
    """Read tertiary contact basepairs from json file"""
    with open(json_file) as f:
        data = json.load(f)
    basepairs = []
    for entry in data:
        bp = Basepair.from_dict(entry["basepair"])
        entry["basepair"] = bp
        basepairs.append(entry)
    return basepairs


def write_interactions_to_cif(motifs, dir_name, pos):
    os.makedirs(os.path.join(dir_name, str(pos)), exist_ok=True)
    for motif in motifs:
        print(motif.name, end=" ")
        motif.to_cif(os.path.join(dir_name, str(pos), f"{motif.name}.cif"))
    print()


def get_hbonds_and_basepairs(pdb_code, all_hbonds, all_basepairs):
    """Get hbonds and basepairs for a given PDB code"""
    hbonds = [hbond for hbond in all_hbonds if hbond["pdb_code"] == pdb_code]
    basepairs = [bp for bp in all_basepairs if bp["pdb_code"] == pdb_code]
    return hbonds, basepairs


def test():
    all_hbonds = get_hbonds_from_json("tertiary_contacts_hbonds.json")
    all_basepairs = get_basepairs_from_json("tertiary_contacts_basepairs.json")
    pdb_codes = get_pdb_ids()
    os.makedirs("tcs", exist_ok=True)
    pos = 0
    for pdb_code in pdb_codes:
        hbonds, basepairs = get_hbonds_and_basepairs(
            pdb_code, all_hbonds, all_basepairs
        )
        if len(hbonds) == 0 and len(basepairs) == 0:
            continue
        # Create dictionary to store interactions between motif pairs
        motif_pair_interactions = {}

        # Process hbonds
        for hbond_entry in hbonds:
            motif1 = hbond_entry["motif1_name"]
            motif2 = hbond_entry["motif2_name"]
            # Sort motif names to ensure consistent key ordering
            key = tuple(sorted([motif1, motif2]))

            if key not in motif_pair_interactions:
                motif_pair_interactions[key] = {"hbonds": [], "basepairs": []}
            motif_pair_interactions[key]["hbonds"].append(hbond_entry["hbond"])

        # Process basepairs
        for bp_entry in basepairs:
            motif1 = bp_entry["motif1_name"]
            motif2 = bp_entry["motif2_name"]
            key = tuple(sorted([motif1, motif2]))

            if key not in motif_pair_interactions:
                motif_pair_interactions[key] = {"hbonds": [], "basepairs": []}
            motif_pair_interactions[key]["basepairs"].append(bp_entry["basepair"])

        motifs_by_name = {}
        # Print summary for each motif pair
        for (motif1, motif2), interactions in motif_pair_interactions.items():
            # Check if interaction counts meet threshold for saving
            if len(interactions["hbonds"]) > 20:
                if "HAIRPIN" in motif1 or "HAIRPIN" in motif2:
                    continue
                # Create directory for this PDB if it doesn't exist
                os.makedirs(os.path.join("tcs", pdb_code), exist_ok=True)

                # Load motifs from JSON
                if len(motifs_by_name) == 0:
                    json_path = os.path.join(
                        DATA_PATH, "jsons", "motifs", f"{pdb_code}.json"
                    )
                    motifs = get_motifs_from_json(json_path)
                    motifs_by_name = {m.name: m for m in motifs}

                write_interactions_to_cif(
                    [motifs_by_name[motif1], motifs_by_name[motif2]], "tcs", pos
                )
                pos += 1
        if pos > 10:
            break


def generate_summary_df():
    df = pd.read_csv("tertiary_contacts_hbonds.csv")
    count = 0
    summary_data = []
    for (motif1, motif2), g in df.groupby(["motif1_name", "motif2_name"]):
        if len(g) < 2:
            continue

        # Count total hbonds and types
        total_hbonds = len(g)
        hbond_types = {}

        # Go through each row and count hbond types based on atom types
        for _, row in g.iterrows():
            atom_type1 = row["atom_type1"]
            atom_type2 = row["atom_type2"]
            # Sort atom types so bb/sugar is same as sugar/bb
            if atom_type1 > atom_type2:
                atom_type1, atom_type2 = atom_type2, atom_type1
            hbond_type = f"{atom_type1}/{atom_type2}"

            if hbond_type not in hbond_types:
                hbond_types[hbond_type] = 0
            hbond_types[hbond_type] += 1

        # Create a row for each motif pair
        row = {"motif1": motif1, "motif2": motif2, "total_hbonds": total_hbonds}

        # Add counts for each hbond type
        for htype, hcount in hbond_types.items():
            row[f"hbond_{htype}"] = hcount

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("tertiary_contacts_summary.csv", index=False)


def write_interactions_to_cif(motifs, dir_name, pos):
    os.makedirs(os.path.join(dir_name, str(pos)), exist_ok=True)
    for motif in motifs:
        print(motif.name, end=" ")
        motif.to_cif(os.path.join(dir_name, str(pos), f"{motif.name}.cif"))
    print()


def main():
    os.makedirs("tcs", exist_ok=True)
    df = pd.read_csv("tertiary_contacts_summary.csv")
    df.sort_values(by="total_hbonds", ascending=True, inplace=True)
    motifs = {}
    count = 0
    for _, row in df.iterrows():
        if row["total_hbonds"] < 10:
            continue
        motif1 = row["motif1"]
        motif2 = row["motif2"]
        pdb_code = motif1.split("-")[0]
        if pdb_code not in motifs:
            json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
            motifs[pdb_code] = {}
            motifs_list = get_motifs_from_json(json_path)
            for motif in motifs_list:
                motifs[pdb_code][motif.name] = motif
        write_interactions_to_cif(
            [motifs[pdb_code][motif1], motifs[pdb_code][motif2]], "tcs", count
        )
        count += 1
        if count > 10:
            break


if __name__ == "__main__":
    main()
