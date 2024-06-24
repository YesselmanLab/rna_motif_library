import os
import subprocess
import settings


# RNP interaction is an interaction between RNA and protein
class RNPInteraction(object):
    def __init__(self, nt_atom, aa_atom, dist, type):
        self.nt_atom = nt_atom  # atom of nucleotide: 'N6@C.A1534'
        self.aa_atom = aa_atom  # atom of amino acid: 'O@A.VAL281'
        self.dist = dist  # distance between the two; we find this on our own
        self.type = type  # type of interaction: 'base:sidechain'
        self.nt_res = nt_atom.split("@")[1]  # residue of nucleotide; 'C.A1534'


# retrieves individual RNP data from .out file
def get_rnp_interactions(pdb_path=None, out_file=None):
    if pdb_path is None and out_file is None:
        raise ValueError("must supply either a pdb or out file")
    """if pdb_path is not None:
        __generate_out_file(pdb_path)
        out_file = 'test.out'"""

    # Open the .out file with RNPs inside and read the lines
    f = open(out_file)
    lines = f.readlines()
    f.close()

    # string that joins the lines
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
            # there is an empty line at the end so filter it out
            if len(i_spl) < 4:
                continue
            # i_spl format: ['1', '8D29', 'OP1@C.U6', 'NH1@H.ARG106', '3.58', 'po4:sidechain:salt-bridge']
            # first need to change the type
            inter_type = i_spl[5]
            inter_type_spl = inter_type.split(":")
            if inter_type_spl[0] == "po4":
                nt_part = "phos"
            elif inter_type_spl[0] == "sugar":
                nt_part = "sugar"
            elif inter_type_spl[0] == "base":
                nt_part = "base"
            inter_type_new = f"{nt_part}:aa"

            interactions.append(RNPInteraction(i_spl[2], i_spl[3], i_spl[4], inter_type_new))

    return interactions


# generates the actual .out file from DSSR data, don't mess with this
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
