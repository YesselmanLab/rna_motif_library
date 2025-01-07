import glob
import os
import json

from pydssr.dssr import DSSROutput

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.classes import X3DNAResidueFactory


def main():
    count = 0
    json_files = glob.glob(os.path.join(DATA_PATH, "dssr_output/*.json"))
    for json_file in json_files:
        print(json_file)
        exit()
        pdb_name = os.path.basename(json_file)[:-5]
        # if "7MQA" not in pdb_name:
        #    continue
        print()
        data = json.load(open(json_file, "r"))
        print(json.dumps(data["nts"][0], indent=4))
        exit()
        for key in data.keys():
            if "HtypePknots" == key:
                print(key, data[key])
                exit()
        continue
        # keys = [k for k in data["dbn"].keys() if k != "all_chains"]
        # for k in keys:
        #    print(k, data["dbn"][k])
        # Get all chain keys except 'all_chains'
        chain_keys = [k for k in data["dbn"].keys() if k != "all_chains"]

        # Get sstr for each chain
        chain_sstrs = {}
        for chain in chain_keys:
            if "sstr" in data["dbn"][chain]:
                chain_sstrs[chain] = data["dbn"][chain]["sstr"]

        # Check for duplicates between chains
        for i, chain1 in enumerate(chain_keys[:-1]):
            if chain1 not in chain_sstrs:
                continue
            for chain2 in chain_keys[i + 1 :]:
                if chain2 not in chain_sstrs:
                    continue
                if chain_sstrs[chain1] == chain_sstrs[chain2]:
                    print(
                        f"{pdb_name}: Duplicate sstr found between chains {chain1} and {chain2}"
                    )
                    print(f"sstr: {chain_sstrs[chain1]}")

        continue
        dssr_output = DSSROutput(json_path=json_file)
        tertiary_contacts = dssr_output.get_tertiary_contacts()
        for c in tertiary_contacts:
            if c.mtype == "RIBOSE_ZIPPER":
                continue
            elif c.mtype == "KISSING_LOOP":
                pass
                # print("kissing loops: ", c.hairpins[0].nts_long)
            else:
                print(c.mtype)
        continue
        dssr_output = DSSROutput(json_path=json_file)
        pairs = dssr_output.get_pairs()
        # contacts = dssr_output.get_tertiary_contacts()
        # for c in contacts:
        #    print(c.__dict__)
        # exit()

        key = list(pairs.keys())[0]
        print(key)
        print(pairs[key].__dict__)
        exit()


if __name__ == "__main__":
    main()
