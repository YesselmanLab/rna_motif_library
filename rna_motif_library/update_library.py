import glob
import requests
import json
import os
import datetime

import settings
import snap
import dssr_lib

from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.pdb.pandas_pdb import PandasPdb

def __safe_mkdir(dir):
    if os.path.isdir(dir):
        return
    os.mkdir(dir)

"""def __download_cif_files(df):
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    count = 0
    for i, row in df.iterrows():
        spl = row[1].split("|")
        pdb_name = spl[0]
        out_path = pdb_dir + f"{pdb_name}.cif"
        path = f"https://files.rcsb.org/download/{pdb_name}.cif"
        if os.path.isfile(out_path):
            count += 1
            print(pdb_name + " ALREADY DOWNLOADED!")
            continue
        else:
            print(pdb_name + " DOWNLOADING")
            wget.download(path, out=out_path)
    print(f"{count} pdbs already downloaded!")"""

def __download_cif_files():
    # Define the directory to save the PDB files
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    # Define the query term
    query_term = {
        "query"          : {
            "type"            : "group",
            "logical_operator": "and",
            "nodes"           : [
                {
                    "type"      : "terminal",
                    "service"   : "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator" : "exact_match",
                        "negation" : False,
                        "value"    : "RNA"
                    }
                },
                {
                    "type"            : "group",
                    "nodes"           : [
                        {
                            "type"      : "terminal",
                            "service"   : "text",
                            "parameters": {
                                "attribute": "exptl.method",
                                "operator" : "exact_match",
                                "negation" : False,
                                "value"    : "X-RAY DIFFRACTION"
                            }
                        },
                        {
                            "type"      : "terminal",
                            "service"   : "text",
                            "parameters": {
                                "attribute": "exptl.method",
                                "operator" : "exact_match",
                                "negation" : False,
                                "value"    : "ELECTRON MICROSCOPY"
                            }
                        }
                    ],
                    "logical_operator": "or"
                }
            ],
            "label"           : "text"
        },
        "return_type"    : "entry",
        "request_options": {
            "paginate"            : {
                "start": 0,
                "rows" : 1000000
            },
            "results_content_type": [
                "experimental"
            ],
            "sort"                : [
                {
                    "sort_by"  : "score",
                    "direction": "desc"
                }
            ],
            "scoring_strategy"    : "combined"
        }
    }
    # Define the API endpoints
    search_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={query_term}"
    download_url = "https://files.rcsb.org/download/"
    # Perform the search and download the PDB files (actually CIF but screw it)
    response = requests.post(search_url, data=json.dumps(query_term))
    results = response.json()["result_set"]
    """# Define the filename to save the response to
    filename = "response.json"
    # Save the response to a file
    with open(filename, "w") as f:
        json.dump(response.json(), f)"""
    for result in results:
        pdb_id = result["identifier"]
        pdb_file = f"{pdb_dir}/{pdb_id}.cif"
        if os.path.exists(pdb_file):
            print(f"{pdb_id} ALREADY DOWNLOADED")
        else:
            pdb_url = f"{download_url}{pdb_id}.cif"
            print(f"{pdb_id} DOWNLOADING")
            response = requests.get(pdb_url)
            with open(pdb_file, "wb") as f:
                f.write(response.content)

    #exit(0)


def __get_dssr_files():
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    dssr_path = settings.DSSR_EXE
    out_path = settings.LIB_PATH + "/data/dssr_output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 0
    for pdb_path in pdbs:
        s = os.path.getsize(pdb_path)
        print(pdb_path, s)  # s = size of file in bytes
        # if s > 10000000:
        #    continue
        name = pdb_path.split("/")[-1][:-4]
        if os.path.isfile(out_path):
            count += 1
            continue
        write_dssr_json_output_to_file(
                dssr_path, pdb_path, out_path + "/" + name + ".out"
        )
        print(out_path + "/" + name + ".out")


def __get_snap_files():
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    out_path = settings.LIB_PATH + "/data/snap_output"

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

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
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    pdbs = glob.glob(pdb_dir + "/*.cif")
    count = 0
    dirs = [
        "motifs",
        "motif_interactions",
        "motifs/twoways",
        "motifs/nways",
        "motif_interactions/twoways",
        "motif_interactions/twoways/all",
        "motif_interactions/nways",
        "motif_interactions/nways/all",
        "motifs/twoways/all",
        "motifs/nways/all",
    ]
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = "motifs/twoways/all"
    interactions_dir = "motif_interactions/twoways/all"
    hbond_vals = [
        ",base:base",
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
    f.write(",".join(hbond_vals) + "\n")
    count = 0
    for pdb_path in pdbs:
        print("Reading " + pdb_path)
        s = os.path.getsize(pdb_path)
        name = pdb_path.split("/")[-1][:-4]
        json_path = settings.LIB_PATH + "/data/dssr_output/" + name + ".out"
        # if s > 10000000:
        #    continue
        print(count, pdb_path)
        count += 1
        try:
            pdb_model = PandasPdb().read_pdb(path=pdb_path)
        except:
            continue
        (
            motifs,
            motif_hbonds,
            motif_interactions,
        ) = dssr_lib.get_motifs_from_structure(json_path)
        for m in motifs:
            print(m.name)
            spl = m.name.split(".")
            if not (spl[0] == "TWOWAY" or spl[0] == "NWAY"):
                continue
            try:
                dssr_lib.write_res_coords_to_pdb(
                        m.nts_long, pdb_model, motif_dir + "/" + m.name
                )
            except:
                continue
            f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")
            if m.name not in motif_hbonds:
                vals = ["0" for _ in hbond_vals]
            else:
                vals = [str(motif_hbonds[m.name][x]) for x in hbond_vals]
            f.write(",".join(vals) + "\n")
            if m.name in motif_interactions:
                try:
                    dssr_lib.write_res_coords_to_pdb(
                            m.nts_long + motif_interactions[m.name],
                            pdb_model,
                            interactions_dir + "/" + m.name + ".inter",
                    )
                except:
                    pass
    f.close()


def main():
    __download_cif_files()
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
    __get_dssr_files()
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
    __get_snap_files()
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
    print("Job finished on", time_string)


if __name__ == '__main__':
    main()
