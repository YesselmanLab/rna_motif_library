import glob
import requests
import json
import os
import datetime
import warnings
from typing import Dict
import pandas as pd

import settings
import snap
import dssr
from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS

operation_counter = 0


# Pandas mmCIF override
class PandasMmcifOverride(PandasMmcif):
    def _construct_df(self, text: str):
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data
        df: Dict[str, pd.DataFrame] = {}
        full_df = pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        # Combine ATOM and HETATM records into the same DataFrame, this solves residue deletion
        combined_df = pd.DataFrame(
                full_df[(full_df.group_PDB == "ATOM") | (full_df.group_PDB == "HETATM")])

        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)

        return combined_df  # Return the combined DataFrame


def __safe_mkdir(dir):
    if os.path.isdir(dir):
        return
    os.mkdir(dir)


def __download_cif_files():
    # Define the directory to save the PDB files
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    # Define the API endpoints
    search_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={settings.QUERY_TERM}"
    download_url = "https://files.rcsb.org/download/"
    # Perform the search and download the PDB files (actually CIF but screw it)
    response = requests.post(search_url, data=json.dumps(settings.QUERY_TERM))
    results = response.json()["result_set"]
    # iterates over each item in the results list obtained from the search response
    for result in results:
        # extracts the PDB identifier (pdb_id) and constructs a file path to save the CIF
        pdb_id = result["identifier"]
        pdb_file = f"{pdb_dir}/{pdb_id}.cif"
        if os.path.exists(pdb_file):
            print(f"{pdb_id} ALREADY DOWNLOADED")
        else:
            pdb_url = f"{download_url}{pdb_id}.cif"
            print(f"{pdb_id} DOWNLOADING")
            response = requests.get(pdb_url)
            # content of the response is then written to the pdb_file
            with open(pdb_file, "wb") as f:
                f.write(response.content)


def __get_dssr_files():
    # creates and sets directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    dssr_path = settings.DSSR_EXE
    out_path = settings.LIB_PATH + "/data/dssr_output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 1
    for pdb_path in pdbs:
        s = os.path.getsize(pdb_path)
        print(count, pdb_path, s)  # s = size of file in bytes
        # if s > 10000000:
        #    continue
        name = pdb_path.split("/")[-1][:-4]
        if os.path.isfile(out_path):
            count += 1
            continue
        # writes raw JSON data
        write_dssr_json_output_to_file(
                dssr_path, pdb_path, out_path + "/" + name + ".json"
        )


def __get_snap_files():
    # creates and sets directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    out_path = settings.LIB_PATH + "/data/snap_output"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # scans every CIF file and stores the output in a .out file
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 0
    for pdb_path in pdbs:
        s = os.path.getsize(pdb_path)
        # if s > 10000000:
        #    continue
        print(count, pdb_path)
        name = pdb_path.split("/")[-1][:-4]
        out_file = out_path + "/" + name + ".out"
        if os.path.isfile(out_file):
            count += 1
            continue
        print(pdb_path)
        snap.__generate_out_file(pdb_path, out_file)
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"


def __generate_motif_files():
    global operation_counter
    # creates directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    pdbs = glob.glob(pdb_dir + "/*.cif")
    dirs = [
        "motifs",
        "motif_interactions",
    ]
    for d in dirs:
        __safe_mkdir(d)
    motif_dir = "motifs/nways/all"
    # opens the file where information about nucleotide interactions are stored
    hbond_vals = [
        "base:base",
        "base:sugar",
        "base:phos",
        "sugar:base",
        "sugar:sugar",
        "sugar:phos",
        "phos:base",
        "phos:sugar",
        "phos:phos",
        "base:aa",
        "sugar:aa",
        "phos:aa",
    ]
    f = open("interactions.csv", "w")
    f.write("name,type,size")
    # writes to the CSV information about nucleotide interactions
    f.write(",".join(hbond_vals) + "\n")
    count = 0
    # writes motif/motif interaction information to PDB files
    for pdb_path in pdbs:
        # debug, here we define which exact pdb path to run
        # if pdb_path == "/Users/jyesselm/PycharmProjects/rna_motif_library/data/pdbs/7URI.cif": # change the part before .cif
        # //debug
        s = os.path.getsize(pdb_path)
        name = pdb_path.split("/")[-1][:-4]
        json_path = settings.LIB_PATH + "/data/dssr_output/" + name + ".json"
        if s < 10000000:  # size-limit on PDB to process, need more computer power to go over this
            count += 1
            print(count, pdb_path, name)
            pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
            (
                motifs,
                motif_hbonds,
                motif_interactions
            ) = dssr.get_motifs_from_structure(json_path)
            for m in motifs:
                print(m.name)
                spl = m.name.split(".")  # this is the filename
                # don't run if there is no way in the name
                if not (spl[0] == "TWOWAY" or spl[0] == "NWAY"):
                    continue

                # Writing to interactions.csv
                f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")
                if m.name not in motif_hbonds:
                    vals = ["0" for _ in hbond_vals]
                else:
                    vals = [str(motif_hbonds[m.name][x]) for x in hbond_vals]
                f.write(",".join(vals) + "\n")
                # if there are no interactions with the motif then it skips and avoids a crash
                try:
                    interactions = motif_interactions[m.name]
                except KeyError:
                    interactions = None  # or any value you want as a default
                # Writing the residues AND interactions to the CIF files
                operation_counter += 1
                dssr.write_res_coords_to_pdb(
                        m.nts_long, interactions, pdb_model,
                        motif_dir + "/" + m.name
                )
                # if operation_counter == 5:
                #    exit(0)

    f.close()
    # dssr.cif_pdb_sort(motif_sort_dir)
    # dssr.cif_pdb_sort(interaction_sort_dir)


def main():
    warnings.filterwarnings("ignore")  # blocks the ragged nested sequence warning
    # time tracking stuff, tracks how long the process takes
    current_time = datetime.datetime.now()
    start_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # start of program
    # __download_cif_files()
    print('''
╔════════════════════════════════════╗
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║       CIF FILES DOWNLOADED         ║
║                                    ║
╚════════════════════════════════════╝
''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    # __get_dssr_files()
    print('''
╔════════════════════════════════════╗
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║       DSSR FILES FINISHED          ║
║                                    ║
╚════════════════════════════════════╝
''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    # __get_snap_files()
    print('''
╔════════════════════════════════════╗
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║       SNAP FILES FINISHED          ║
║                                    ║
╚════════════════════════════════════╝
''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job finished on", time_string)
    __generate_motif_files()
    print('''
╔════════════════════════════════════╗
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║                                    ║
║      MOTIF FILES FINISHED          ║
║                                    ║
╚════════════════════════════════════╝
''')
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")  # format time as string
    print("Job started on", start_time_string)
    print("Job finished on", time_string)


if __name__ == '__main__':
    main()
