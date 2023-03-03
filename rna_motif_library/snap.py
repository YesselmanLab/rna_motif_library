import os
import subprocess
import settings

class RNPInteraction(object):
    def __init__(self, nt_atom, aa_atom, dist, type):
        self.nt_atom = nt_atom
        self.aa_atom = aa_atom
        self.dist = dist
        self.type = type
        self.nt_res = nt_atom.split("@")[1]

def get_rnp_interactions(pdb_path=None, out_file=None):
    if pdb_path is None and out_file is None:
        raise ValueError("must supply either a pdb or out file")
    if pdb_path is not None:
        __generate_out_file(pdb_path)
        out_file = 'test.out'
    f = open(out_file)
    lines = f.readlines()
    f.close()
    s = "".join(lines)
    spl = s.split("List")
    interactions = []
    for s in spl:
        if s.find('H-bonds') == -1:
            continue
        lines = s.split("\n")
        lines.pop(0)
        lines.pop(0)
        for l in lines:
            i_spl = l.split()
            if len(i_spl) < 4:
                continue
            interactions.append(RNPInteraction(i_spl[2], i_spl[3], i_spl[4], i_spl[5]))
    return interactions

def __generate_out_file(pdb_path, out_path="test.out"):
    dssr_exe = settings.DSSR_EXE
    subprocess.run(f"{dssr_exe} snap -i={pdb_path} -o={out_path}", shell=True)
    files = "dssr-2ndstrs.bpseq,dssr-2ndstrs.ct,dssr-2ndstrs.dbn,dssr-atom2bases.pdb,dssr-stacks.pdb,dssr-torsions.txt".split(
        ",")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


