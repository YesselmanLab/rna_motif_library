import os
import json
import pickle
import pandas as pd
from multiprocessing import Pool
from functools import partial
from pydssr.dssr import DSSROutput
from rna_motif_library.classes import Basepair
from rna_motif_library.cli import get_pdb_ids
from rna_motif_library.interactions import get_hbonds_and_basepairs
from rna_motif_library.motif import (
    get_motifs_from_json,
    MotifFactory,
    are_residues_connected,
)
from rna_motif_library.settings import DATA_PATH


def process_pdb_chunk(pdb_codes):
    motifs = []
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
        if os.path.exists(json_path):
            motifs.append(get_motifs_from_json(json_path))
    return [m for m in motifs if m is not None]


def load_motifs():
    """
    Load motifs from pickle file.

    Returns:
        list: List of motifs loaded from pickle file
    """
    pickle_path = "motifs.pkl"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_contained_motifs():
    """
    Load contained motifs from pickle file.

    Returns:
        list: List of contained motif pairs loaded from pickle file
    """
    pickle_path = "contained_motifs.pkl"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def create_motif_pickle():
    pdb_codes = get_pdb_ids()
    num_processes = 15  # Default number of processes
    chunk_size = len(pdb_codes) // num_processes
    if len(pdb_codes) % num_processes:
        chunk_size += 1

    chunks = [
        pdb_codes[i : i + chunk_size] for i in range(0, len(pdb_codes), chunk_size)
    ]

    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_pdb_chunk, chunks)

    pdb_motifs = [motif for chunk in chunk_results for motif in chunk]

    # Save motifs to pickle file
    pickle_path = "motifs.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(pdb_motifs, f)


def get_contained_motifs():
    all_motifs = load_motifs()
    os.makedirs("overlaps", exist_ok=True)

    contained_motifs = []
    for motifs in all_motifs:
        for i, motif1 in enumerate(motifs):
            for j, motif2 in enumerate(motifs):
                if i >= j:
                    continue
                for strand1 in motif1.strands:
                    for strand2 in motif2.strands:
                        if all(res in strand1 for res in strand2):
                            print(motif1.name, motif2.name)
                            contained_motifs.append((motif1, motif2))
                            break
                        else:
                            continue

    print(len(contained_motifs))
    for pos, (motif1, motif2) in enumerate(contained_motifs):
        write_interactions_to_cif([motif1, motif2], "overlaps", pos)
    # Save contained motifs to pickle file
    with open("contained_motifs.pkl", "wb") as f:
        pickle.dump(contained_motifs, f)
    exit()

    return contained_motifs


def count_non_basepair_overlaps(motifs):
    """Count residue overlaps between motifs that aren't part of shared basepairs.

    Args:
        motifs (List[Motif]): List of motifs to analyze

    Returns:
        Dict[str, Dict]: Dictionary mapping motif names to overlap info containing:
            - 'total_overlaps': Total number of overlapping residues not in basepairs
            - 'overlapping_motifs': Dict mapping overlapping motif names to overlap counts
            - 'overlapping_residues': Dict mapping overlapping motif names to lists of overlapping residues
    """
    overlap_info = {}

    for i, motif1 in enumerate(motifs):
        overlap_info[motif1.name] = {
            "total_overlaps": 0,
            "overlapping_motifs": {},
            "overlapping_residues": {},
        }

        # Get set of residues involved in basepairs for motif1
        motif1_bp_residues = set()
        for bp in motif1.basepair_ends:
            motif1_bp_residues.add(bp.res_1.get_str())
            motif1_bp_residues.add(bp.res_2.get_str())

        # Get all residues for motif1 not in basepairs
        motif1_non_bp_residues = set()
        for res in motif1.get_residues():
            if res.get_x3dna_str() not in motif1_bp_residues:
                motif1_non_bp_residues.add(res.get_x3dna_str())

        for j, motif2 in enumerate(motifs):
            if i >= j:
                continue

            # Get set of residues involved in basepairs for motif2
            motif2_bp_residues = set()
            for bp in motif2.basepair_ends:
                motif2_bp_residues.add(bp.res_1.get_str())
                motif2_bp_residues.add(bp.res_2.get_str())

            # Get all residues for motif2 not in basepairs
            motif2_non_bp_residues = set()
            for res in motif2.get_residues():
                if res.get_x3dna_str() not in motif2_bp_residues:
                    motif2_non_bp_residues.add(res.get_x3dna_str())

            # Find overlap between non-basepair residues
            overlapping_residues = motif1_non_bp_residues.intersection(
                motif2_non_bp_residues
            )
            overlap = len(overlapping_residues)

            if overlap > 0:
                overlap_info[motif1.name]["total_overlaps"] += overlap
                overlap_info[motif1.name]["overlapping_motifs"][motif2.name] = overlap
                overlap_info[motif1.name]["overlapping_residues"][motif2.name] = list(
                    overlapping_residues
                )

                # Add reciprocal info for motif2
                if motif2.name not in overlap_info:
                    overlap_info[motif2.name] = {
                        "total_overlaps": 0,
                        "overlapping_motifs": {},
                        "overlapping_residues": {},
                    }
                overlap_info[motif2.name]["total_overlaps"] += overlap
                overlap_info[motif2.name]["overlapping_motifs"][motif1.name] = overlap
                overlap_info[motif2.name]["overlapping_residues"][motif1.name] = list(
                    overlapping_residues
                )

    return overlap_info


def extract_hairpin_motif(hairpin_motif, other_motif):
    # Find the hairpin strand in the other motif
    hairpin_strand = None
    for strand in other_motif.strands:
        # Check if this strand is contained within any of the hairpin motif's strands
        for hairpin_strand_candidate in hairpin_motif.strands:
            if all(res in hairpin_strand_candidate for res in strand):
                hairpin_strand = strand
                break
        if hairpin_strand is not None:
            break

    residues = []
    for strand in other_motif.strands:
        if strand != hairpin_strand:
            residues.extend(strand)

    mf = MotifFactory(other_motif.pdb, other_motif.hbonds, other_motif.basepairs)
    new_motif = mf.from_residues(residues)
    return new_motif


def write_interactions_to_cif(motifs, dir_name, pos):
    os.makedirs(os.path.join(dir_name, str(pos)), exist_ok=True)
    for motif in motifs:
        print(motif.name, end=" ")
        motif.to_cif(os.path.join(dir_name, str(pos), f"{motif.name}.cif"))
    print()


def find_pair():
    dssr_output = DSSROutput(
        json_path=os.path.join(DATA_PATH, "dssr_output", "6RM3.json")
    )
    pairs = dssr_output.get_pairs()
    for pair in pairs.values():
        if pair.nt1.nt_id == "S60.U11" or pair.nt2.nt_id == "S60.U11":
            print(pair.nt1.nt_id, pair.nt2.nt_id)


def are_motifs_connected(motif_1, motif_2):
    # Check each strand's end residues for connections
    for strand1 in motif_1.strands:
        for strand2 in motif_2.strands:
            # Get end residues
            strand1_ends = [strand1[0], strand1[-1]]
            strand2_ends = [strand2[0], strand2[-1]]

            # Check all combinations of end residues
            for res1 in strand1_ends:
                for res2 in strand2_ends:
                    if are_residues_connected(res1, res2) != 0:
                        return True
    return False


def are_motifs_sequential(motif_1, motif_2):
    """
    Check if any strands between two motifs are sequential (connected in sequence).

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if any strands are sequential, False otherwise
    """
    # Check each strand's end residues for sequential connections
    for strand1 in motif_1.strands:
        for strand2 in motif_2.strands:
            # Get end residues
            strand1_ends = [strand1[0], strand1[-1]]
            strand2_ends = [strand2[0], strand2[-1]]

            # Check all combinations of end residues
            for res1 in strand1_ends:
                for res2 in strand2_ends:
                    # Check if residues are on same chain
                    if res1.chain_id != res2.chain_id:
                        continue

                    # Check if residue numbers differ by 1
                    if abs(res1.num - res2.num) == 1:
                        return True

    return False


def have_common_basepair(motif_1, motif_2):
    """
    Check if two motifs share any common basepairs.

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if motifs share any basepairs, False otherwise
    """
    # Get all basepairs from both motifs
    bps_1 = set((bp.res_1.get_str(), bp.res_2.get_str()) for bp in motif_1.basepairs)
    bps_2 = set((bp.res_1.get_str(), bp.res_2.get_str()) for bp in motif_2.basepairs)

    # Also check reverse order of residues
    bps_1.update((bp.res_2.get_str(), bp.res_1.get_str()) for bp in motif_1.basepairs)
    bps_2.update((bp.res_2.get_str(), bp.res_1.get_str()) for bp in motif_2.basepairs)

    # Check for any common basepairs
    return len(bps_1.intersection(bps_2)) > 0


def check_residue_overlap(motif_1, motif_2):
    """
    Check if two motifs have any overlapping residues.

    Args:
        motif_1: First motif to check
        motif_2: Second motif to check

    Returns:
        bool: True if motifs have overlapping residues, False otherwise
    """
    # Get all residues from both motifs
    residues_1 = set(res.get_x3dna_str() for res in motif_1.get_residues())
    residues_2 = set(res.get_x3dna_str() for res in motif_2.get_residues())

    # Check for any overlapping residues
    return len(residues_1.intersection(residues_2)) > 0


def find_tertiary_contact_interactions():
    os.makedirs("tcs", exist_ok=True)
    pdb_codes = get_pdb_ids()
    tcs_hbonds = []
    tcs_basepairs = []
    count = 0
    csv_hbond_data = []
    csv_bp_data = []
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
        if not os.path.exists(json_path):
            continue
        motifs = get_motifs_from_json(json_path)
        motifs_by_name = {m.name: m for m in motifs}
        motif_res = {}
        motif_res_pairs = {}
        for motif in motifs:
            for res in motif.get_residues():
                motif_res[res.get_x3dna_str()] = motif.name
                motif_res_pairs[motif.name + "-" + res.get_x3dna_str()] = True
        hbonds, basepairs = get_hbonds_and_basepairs(pdb_code)
        for hbond in hbonds:
            if hbond.hbond_type == "RNA/PROTEIN":
                continue
            if (
                hbond.res_1.get_str() not in motif_res
                or hbond.res_2.get_str() not in motif_res
            ):
                continue
            motif_1_name = motif_res[hbond.res_1.get_str()]
            motif_2_name = motif_res[hbond.res_2.get_str()]
            key_1 = motif_2_name + "-" + hbond.res_1.get_str()
            key_2 = motif_1_name + "-" + hbond.res_2.get_str()
            if key_1 in motif_res_pairs or key_2 in motif_res_pairs:
                continue
            if motif_1_name == motif_2_name:
                continue
            if "UNKNOWN" in motif_1_name or "UNKNOWN" in motif_2_name:
                continue
            if are_motifs_connected(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if are_motifs_sequential(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if have_common_basepair(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if check_residue_overlap(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            tcs_hbonds.append([pdb_code, motif_1_name, motif_2_name, hbond])
            csv_hbond_data.append(
                {
                    "pdb_code": pdb_code,
                    "motif1_name": motif_1_name,
                    "motif2_name": motif_2_name,
                    "res1": hbond.res_1.get_str(),
                    "res2": hbond.res_2.get_str(),
                    "atom1": hbond.atom_1,
                    "atom2": hbond.atom_2,
                    "atom_type1": hbond.atom_type_1,
                    "atom_type2": hbond.atom_type_2,
                    "distance": hbond.distance,
                    "angle": hbond.angle,
                    "hbond_type": hbond.hbond_type,
                }
            )
        for bp in basepairs:
            if (
                bp.res_1.get_str() not in motif_res
                or bp.res_2.get_str() not in motif_res
            ):
                continue
            motif_1_name = motif_res[bp.res_1.get_str()]
            motif_2_name = motif_res[bp.res_2.get_str()]
            key_1 = motif_2_name + "-" + bp.res_1.get_str()
            key_2 = motif_1_name + "-" + bp.res_2.get_str()
            if key_1 in motif_res_pairs or key_2 in motif_res_pairs:
                continue
            if motif_1_name == motif_2_name:
                continue
            if "UNKNOWN" in motif_1_name or "UNKNOWN" in motif_2_name:
                continue
            if are_motifs_connected(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if are_motifs_sequential(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if have_common_basepair(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            if check_residue_overlap(
                motifs_by_name[motif_1_name], motifs_by_name[motif_2_name]
            ):
                continue
            tcs_basepairs.append([pdb_code, motif_1_name, motif_2_name, bp])

    # Write tertiary contact hbonds to json file
    output = []
    for pdb_code, motif1_name, motif2_name, hbond in tcs_hbonds:
        output.append(
            {
                "pdb_code": pdb_code,
                "motif1_name": motif1_name,
                "motif2_name": motif2_name,
                "hbond": hbond.to_dict(),
            }
        )
    with open("tertiary_contacts_hbonds.json", "w") as f:
        json.dump(output, f)

    # Write tertiary contact basepairs to json file
    output = []
    for pdb_code, motif1_name, motif2_name, bp in tcs_basepairs:
        output.append(
            {
                "pdb_code": pdb_code,
                "motif1_name": motif1_name,
                "motif2_name": motif2_name,
                "basepair": bp.to_dict(),
            }
        )
    with open("tertiary_contacts_basepairs.json", "w") as f:
        json.dump(output, f)
    df = pd.DataFrame(csv_hbond_data)
    df.to_csv("tertiary_contacts_hbonds.csv", index=False)


def find_rna_protein_interactions():
    pdb_codes = get_pdb_ids()
    count = 0
    csv_hbond_data = []
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
        if not os.path.exists(json_path):
            continue
        hbonds, _ = get_hbonds_and_basepairs(pdb_code)
        for hbond in hbonds:
            if hbond.hbond_type != "RNA/PROTEIN":
                continue
            if hbond.atom_type_1 == "aa":
                res1, res2 = hbond.res_2, hbond.res_1
                atom1, atom2 = hbond.atom_2, hbond.atom_1
                atom_type1, atom_type2 = hbond.atom_type_2, hbond.atom_type_1
                res1_id, res2_id = hbond.res_2.res_id, hbond.res_1.res_id
            else:
                res1, res2 = hbond.res_1, hbond.res_2
                atom1, atom2 = hbond.atom_1, hbond.atom_2
                atom_type1, atom_type2 = hbond.atom_type_1, hbond.atom_type_2
                res1_id, res2_id = hbond.res_1.res_id, hbond.res_2.res_id
            csv_hbond_data.append(
                {
                    "pdb_code": pdb_code,
                    "res1_id": res1_id,
                    "res2_id": res2_id,
                    "atom1": atom1,
                    "atom2": atom2,
                    "atom_type1": atom_type1,
                    "atom_type2": atom_type2,
                    "distance": hbond.distance,
                    "angle": hbond.angle,
                    "res1": res1.get_str(),
                    "res2": res2.get_str(),
                }
            )
    df = pd.DataFrame(csv_hbond_data)
    df.to_csv("rna_protein_interactions.csv", index=False)


def process_motifs():
    count = 0
    pdb_codes = get_pdb_ids()
    # df_rna_counts = pd.read_csv("rna_residue_counts.csv")
    # df_rna_counts = df_rna_counts[df_rna_counts["count"] < 200]
    # pdb_codes = df_rna_counts["pdb_code"].to_list()
    data = []
    total_motifs = 0
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
        if not os.path.exists(json_path):
            continue
        motifs = get_motifs_from_json(json_path)
        motif_res = {}
        for motif in motifs:
            for res in motif.get_residues():
                motif_res[res.get_x3dna_str()] = motif.name

        count = 0
        for motif in motifs:
            if motif.mtype != "UNKNOWN":
                continue

            # if len(motif.get_residues()) < 30:
            #    continue

            total_motifs += 1
            count += 1
        print(pdb_code, count)
        data.append({"pdb_code": pdb_code, "count": count})
    print(total_motifs)
    df = pd.DataFrame(data)
    df.to_csv("unknown_motifs.csv", index=False)


def get_bp_type(bp) -> str:
    return bp.res_1.res_id + "-" + bp.res_2.res_id


def process_motifs_2():
    all_motifs = load_motifs()
    total_motifs = 0
    seen = []
    for motifs in all_motifs:
        for motif in motifs:
            for bp in motif.basepair_ends:
                if get_bp_type(bp) not in seen:
                    seen.append(get_bp_type(bp))
                    try:
                        motif.to_cif(os.path.join("motifs", f"{motif.name}.cif"))
                    except Exception as e:
                        print(e)
    f = open("end_pairs.txt", "w")
    for bp_name in seen:
        f.write(bp_name + "\n")
    f.close()


def find_overlapping_motifs():
    pdb_codes = get_pdb_ids()
    pos = 0
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
        if not os.path.exists(json_path):
            continue

        motifs = get_motifs_from_json(json_path)
        motif_dict = {motif.name: motif for motif in motifs}
        overlap_info = count_non_basepair_overlaps(motifs)

        for key, value in overlap_info.items():
            if value["total_overlaps"] > 10:
                print(f"{pdb_code} {key}", value)
                all_motifs = []
                for motif_name in value["overlapping_motifs"]:
                    all_motifs.append(motif_dict[motif_name])
                all_motifs.append(motif_dict[key])
                write_interactions_to_cif(all_motifs, "overlaps", pos)
                pos += 1


def main():
    process_motifs_2()
    exit()
    df = pd.read_csv("unknown_motifs.csv")
    df = df.sort_values(by="count", ascending=False)
    print(df.head(10))

    exit()
    all_motifs = load_motifs()
    os.makedirs("overlaps", exist_ok=True)
    count = 0
    for motifs in all_motifs:
        for i, motif1 in enumerate(motifs):
            for j, motif2 in enumerate(motifs):
                if i >= j:
                    continue
                # Count residue overlap between motifs
                overlap_count = 0
                for res1 in motif1.get_residues():
                    for res2 in motif2.get_residues():
                        if res1 == res2:
                            overlap_count += 1
            if overlap_count > 2:
                os.makedirs(os.path.join("overlaps", str(count)), exist_ok=True)
                print(motif1.name, motif2.name, overlap_count)
                motif1.to_cif(
                    os.path.join("overlaps", str(count), f"{motif1.name}.cif")
                )
                motif2.to_cif(
                    os.path.join("overlaps", str(count), f"{motif2.name}.cif")
                )
                count += 1
    print(count)
    exit()
    count = 0
    contained_motifs = load_contained_motifs()
    contained_motifs_dict = {}

    for motif1, motif2 in contained_motifs:
        if not ("HAIRPIN" in motif1.name or "HAIRPIN" in motif2.name):
            continue
        if "HAIRPIN" in motif1.name:
            hairpin_motif = motif1
            other_motif = motif2
        else:
            hairpin_motif = motif2
            other_motif = motif1
        os.makedirs(os.path.join("overlaps", str(count)), exist_ok=True)
        other_motif.to_cif(
            os.path.join("overlaps", str(count), f"{other_motif.name}.cif")
        )
        hairpin_motif.to_cif(
            os.path.join("overlaps", str(count), f"{hairpin_motif.name}.cif")
        )
        new_motif = extract_hairpin_motif(hairpin_motif, other_motif)
        new_motif.to_cif(os.path.join("overlaps", str(count), f"new.cif"))
        count += 1


if __name__ == "__main__":
    main()
