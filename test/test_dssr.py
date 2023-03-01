import rna_motif_library.dssr
import rna_motif_library.settings
from biopandas.pdb.pandas_pdb import PandasPdb


def test_get_motifs_from_structure():
    json_path = rna_motif_library.settings.UNITTEST_PATH + "/resources/1GID.json"
    pdb_path = rna_motif_library.settings.UNITTEST_PATH + "/resources/1GID.pdb"
    motifs = rna_motif_library.dssr.get_motifs_from_structure(json_path)
    rna_motif_library.dssr.write_motif_coords_to_pdbs(motifs, pdb_path) # fix this, it doesn't exist anymore

def test_dssr_res():
    s1 = 'H.A9'
    s2 = 'B.ARG270'
    r1 = rna_motif_library.dssr.DSSRRes(s1)
    assert r1.res_id == 'A'
    assert r1.chain_id == 'H'
    assert r1.num == 9
    r2 = rna_motif_library.dssr.DSSRRes(s2)
    assert r2.res_id == 'ARG'
    assert r2.num == 270

def test_from_lib():
    name = '5WT1'
    pdb_path = rna_motif_library.settings.LIB_PATH + "/data/pdbs/" + name + ".cif"
    json_path = rna_motif_library.settings.LIB_PATH + "/data/dssr_output/" + name + ".out"
    motifs, motif_hbonds, motif_interactions = rna_motif_library.dssr.get_motifs_from_structure(json_path)
    pdb_model = PandasPdb.read_pdb(pdb_path) # this one is an argument bitch for some reason
    for m in motifs:
        if m.name not in motif_interactions:
            interactions = []
        else:
            interactions = motif_interactions[m.name]
        if m.name in motif_hbonds:
            print(m.name, motif_hbonds[m.name])
        rna_motif_library.dssr.write_res_coords_to_pdb(m.nts_long, pdb_model, m.name) #fix this it doesn't exist anymore
        if len(interactions) > 0:
            rna_motif_library.dssr.write_res_coords_to_pdb(m.nts_long + interactions, pdb_model, m.name + ".inter") #fix this, it doesn't exist anymore


'''

def _test_motifs_to_pdbs():
    json_path = '1GID.json'
    exit()
    pdb_path = '1GID.pdb'
    d_out = dssr.DSSROutput(json_path='1GID.json')
    cif1 = biopandas.PandasPdb().read_pdb(pdb_path)
    motifs = d_out.get_motifs()
    count = 0
    for m in motifs:
        res = []
        for nt in m.nts_long:
            spl = nt.split(".")
            new_nt = spl[0] + "." + spl[1][1:]
            res.append(cif1.model.residue(new_nt))
        s = ""
        for r in res:
           lines = structure_to_pdb_string(r).split("\n")
           s += "\n".join(lines[:-1]) + "\n"
        f = open(f"{m.mtype}.{count}.pdb", "w")
        count += 1
        f.write(s)
        f.close()
        
'''


def main():
    test_from_lib()


if __name__ == "__main__":
    main()
