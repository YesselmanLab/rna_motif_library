import pandas as pd
import wget
import os
import glob

import settings
import snap
import dssr_lib

from pydssr.dssr import write_dssr_json_output_to_file
from biopandas.pdb.pandas_pdb import PandasPdb
from flask import Flask, jsonify

app = Flask(__name__)


def __safe_mkdir(dir):
    if os.path.isdir(dir):
        return
    os.mkdir(dir)


def __download_cif_files(df):
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
            # Check if the PDB file has RNA chains
            #if "RNA" not in row["macromolecules"]:
            #    print(pdb_name + " SKIPPED - NO RNA CHAINS")
            #    continue
            wget.download(path, out=out_path)
    print(f"{count} pdbs already downloaded!")


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
        print(pdb_path, s)
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

# just grab any PDB w/ RNA in it
# use RestAPI, ask Eric (Erik?)

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


@app.route('/update_library')
def update_RNA_library():
    csv_path = settings.LIB_PATH + "/data/csvs/nrlist_3.189_3.5A.csv"
    df = pd.read_csv(csv_path)
    __download_cif_files(df)
    print("CIF files downloaded")
    __get_dssr_files()
    print("DSSR files generated")
    __get_snap_files()
    print("SNAP files generated")
    __generate_motif_files()
    print("Motif files generated")

    response_data = {'message': 'Library updated'}
    response = jsonify(response_data)
    return response

def main():
    ##### one CSV
    csv_path = settings.LIB_PATH + "/data/csvs/nrlist_3.189_3.5A.csv"
    print(csv_path)
    df = pd.read_csv(csv_path)
    __download_cif_files(df)
    __get_dssr_files()
    __get_snap_files()
    __generate_motif_files()
    ##### all CSVs in the folder
    '''
    csv_dir = settings.LIB_PATH + "/data/csvs/"
    csv_files = glob.glob(csv_dir + "*.csv")
    for csv_path in csv_files:
        print(csv_path)
        df = pd.read_csv(csv_path)
        __download_cif_files(df)
        __get_dssr_files()
        __get_snap_files()
        __generate_motif_files()'''
    #exit(0)



if __name__ == '__main__':
    app.run()
