import pandas as pd
import click
import os
from typing import List, Dict
from dataclasses import dataclass

from rna_motif_library.util import get_pdb_ids
from rna_motif_library.x3dna import X3DNAResidue
from rna_motif_library.chain import get_rna_chains, Chains, write_chain_to_cif
from rna_motif_library.motif import (
    MotifFactory,
    Motif,
    save_motifs_to_json,
    get_cached_motifs,
    get_motifs_from_json,
)
from rna_motif_library.basepair import Basepair, get_cached_basepairs
from rna_motif_library.residue import Residue, get_cached_residues
from rna_motif_library.resources import ResidueManager
from rna_motif_library.settings import DATA_PATH


@dataclass
class ResidueId:
    pdb_id: str
    model: str
    chain: str
    residue_type: str
    residue_num: int
    insert_code: str

    @staticmethod
    def _parse_triple_pipe_string(residue_str: str):
        spl_1 = residue_str.split("|||")
        if len(spl_1) != 1:
            insert_code = spl_1[1]
            # dont understand why this is sometimes not an insertion code
            if len(insert_code) > 1:
                insert_code = ""
        else:
            insert_code = ""

    @classmethod
    def from_string(cls, residue_str: str):
        # Split on | delimiter
        pdb_id, model, chain, residue_type, residue_num = residue_str.split("|")[0:5]
        insert_code = cls._parse_triple_pipe_string(residue_str)
        spl_2 = residue_str.split("||")
        if insert_code != "" and len(spl_2) != 1:
            if len(spl_2[1]) > 1:
                insert_code = spl_2[1][-1]
            else:
                insert_code = ""
        else:
            insert_code = ""
        return cls(
            pdb_id=pdb_id,
            model=model,
            chain=chain,
            residue_type=residue_type,
            residue_num=int(residue_num),
            insert_code=insert_code,
        )

    def to_x3dna_residue(self) -> X3DNAResidue:
        return X3DNAResidue(
            chain_id=self.chain,
            res_id=self.residue_type,
            num=self.residue_num,
            ins_code=self.insert_code,
            rtype=self.residue_type,
        )


@click.group()
def cli():
    """Command line interface for getting motifs from atlas CSV."""
    pass


def parse_atlas_csv(csv_path):
    f = open(csv_path, "r")
    group = None
    mtype = None
    data = []
    if os.path.basename(csv_path).startswith("hl"):
        mtype = "HAIRPIN"
    elif os.path.basename(csv_path).startswith("il"):
        mtype = "INTERNAL_LOOP"
    elif os.path.basename(csv_path).startswith("j3"):
        mtype = "NWAY"
    else:
        raise ValueError(f"Unknown motif type: {os.path.basename(csv_path)}")
    for line in f:
        if line.startswith(">"):
            group = line.strip()[1:]
            continue
        residue_infos = line.strip().split(",")
        x3dna_res = []
        pdb_id = None
        for residue_info in residue_infos:
            # remove quotes
            residue_info = residue_info[1:-1]
            residue_id = ResidueId.from_string(residue_info)
            x3dna_res.append(residue_id.to_x3dna_residue().get_str())
            pdb_id = residue_id.pdb_id
        data.append(
            {
                "pdb_id": pdb_id,
                "group": group,
                "mtype": mtype,
                "residues": x3dna_res,
            }
        )
    return pd.DataFrame(data)


@cli.command()
@click.argument("csv_paths", type=click.Path(exists=True), nargs=-1)
def collect_motifs(csv_paths):
    """Collect motifs from CSV files into a single file."""
    dfs = []
    for csv_path in csv_paths:
        dfs.append(parse_atlas_csv(csv_path))
    df = pd.concat(dfs)
    df.to_json("atlas_motifs.json", orient="records")


def get_residues_for_motif(residues: Dict[str, Residue], x3dna_res: List[str]):
    return [residues[x3dna_res] for x3dna_res in x3dna_res]


def do_number_of_strands_match_mtype(mtype: str, strands: List[List[Residue]]):
    if mtype == "HAIRPIN":
        return len(strands) == 1
    elif mtype == "INTERNAL_LOOP":
        return len(strands) == 2
    elif mtype == "NWAY":
        return len(strands) == 3


@cli.command()
@click.argument("json_path", type=click.Path(exists=True))
def get_motifs(json_path):
    pdb_ids = get_pdb_ids()
    df = pd.read_json(json_path)
    df["is_valid"] = "YES"
    for pdb_id in pdb_ids:
        print(pdb_id)
        motifs = []
        residues = get_cached_residues(pdb_id)
        basepairs = get_cached_basepairs(pdb_id)
        chains = get_rna_chains(residues.values())
        mf = MotifFactory(pdb_id, Chains(chains), basepairs)
        df_pdb = df[df["pdb_id"] == pdb_id]
        for _, row in df_pdb.iterrows():
            try:
                motif_residues = get_residues_for_motif(residues, row["residues"])
            except:
                continue
            chains = get_rna_chains(motif_residues)
            is_valid = do_number_of_strands_match_mtype(row["mtype"], chains)
            if not is_valid:
                df.loc[row.name, "is_valid"] = "NO"
                continue
            mtype = row["mtype"]
            if mtype == "INTERNAL_LOOP":
                mtype = "TWOWAY"
            m = Motif(
                "",
                mtype,
                pdb_id,
                "",
                "",
                chains,
                [],
                [],
                [],
            )
            m = mf._finalize_motif(m)
            motifs.append(m)
        save_motifs_to_json(motifs, f"{DATA_PATH}/atlas_motifs/{pdb_id}.json")
    df.to_json("atlas_motifs_validated.json", orient="records")


@cli.command()
def compare_motifs():
    pdb_ids = get_pdb_ids()
    pdb_ids = ["8C3A"]
    data = []
    for pdb_id in pdb_ids:
        print(pdb_id)
        motifs = get_cached_motifs(pdb_id)
        atlas_motif_path = f"{DATA_PATH}/atlas_motifs/{pdb_id}.json"
        atlas_motifs = get_motifs_from_json(atlas_motif_path)
        motifs_by_name = {motif.name: motif for motif in motifs}
        atlas_motifs_by_name = {motif.name: motif for motif in atlas_motifs}
        for motif_name in motifs_by_name.keys():
            m = motifs_by_name[motif_name]
            # atlas does not do more than 3-way junctions
            if len(m.basepair_ends) > 3:
                continue
            # atlas does not do s-strands or helices
            if m.mtype == "SSTRAND" or m.mtype == "HELIX":
                continue
            if motif_name not in atlas_motifs_by_name:
                data.append(
                    {
                        "pdb_id": pdb_id,
                        "motif_name": motif_name,
                        "motif_type": m.mtype,
                        "motif_size": m.num_residues(),
                        "missed_by": "atlas",
                    }
                )
        for motif_name in atlas_motifs_by_name.keys():
            if motif_name not in motifs_by_name:
                data.append(
                    {
                        "pdb_id": pdb_id,
                        "motif_name": motif_name,
                        "motif_type": atlas_motifs_by_name[motif_name].mtype,
                        "motif_size": atlas_motifs_by_name[motif_name].num_residues(),
                        "missed_by": "motif_library",
                    }
                )
    df = pd.DataFrame(data)
    df.to_json("missed_motifs.json", orient="records")


@cli.command()
def check_missing_motifs():
    df = pd.read_json("missed_motifs.json")
    df = df[df["missed_by"] == "motif_library"]
    print(len(df))
    for i, group in df.groupby("pdb_id"):
        if len(group) < 10:
            continue
        print(i, len(group))


@cli.command()
@click.argument("pdb_id", type=str)
@click.argument("res_num_1", type=int)
@click.argument("res_num_2", type=int)
def inspect_basepair(pdb_id, res_num_1, res_num_2):
    basepairs = get_cached_basepairs(pdb_id)
    for bp in basepairs:
        if bp.res_1.num == res_num_1 and bp.res_2.num == res_num_2:
            print(bp.bp_type, bp.hbond_score, len(bp.hbonds))
        elif bp.res_1.num == res_num_2 and bp.res_2.num == res_num_1:
            print(bp.bp_type, bp.hbond_score, len(bp.hbonds))
    exit()


@cli.command()
@click.argument("pdb_id", type=str)
def inspect_missing_motifs(pdb_id):
    os.makedirs(f"overlaps", exist_ok=True)
    pos = 0
    motifs = get_cached_motifs(pdb_id)
    atlas_motifs = get_motifs_from_json(f"{DATA_PATH}/atlas_motifs/{pdb_id}.json")
    motifs_by_name = {motif.name: motif for motif in motifs}
    atlas_motifs_by_name = {motif.name: motif for motif in atlas_motifs}
    basepairs = get_cached_basepairs(pdb_id)
    residues = get_cached_residues(pdb_id)
    chains = get_rna_chains(residues.values())
    mf = MotifFactory(pdb_id, Chains(chains), basepairs)
    df = pd.read_json("missed_motifs.json")
    df = df[df["pdb_id"] == pdb_id]
    df = df[df["missed_by"] == "motif_library"]
    for _, row in df.iterrows():
        atlas_motif = atlas_motifs_by_name[row["motif_name"]]
        residues = atlas_motif.get_residues()
        residue_dict = {residue.get_x3dna_str(): residue for residue in residues}
        count = 0
        overlap_motifs = []
        for motif in motifs:
            if motif.mtype == "HELIX":
                continue
            motif_residues = motif.get_residues()
            motif_residue_overlap = 0
            for motif_residue in motif_residues:
                if motif_residue.get_x3dna_str() in residue_dict:
                    motif_residue_overlap += 1
            if motif_residue_overlap == 0:
                continue
            has_singlet = False
            for bp in motif.basepairs:
                bp_str = bp.res_1.get_x3dna_str() + "-" + bp.res_2.get_x3dna_str()
                if bp_str in mf.singlet_pairs_lookup:
                    has_singlet = True
                    break
            if not has_singlet and motif_residue_overlap >= len(residues):
                continue
            print(motif.name, len(motif_residues), len(residues))
            overlap_motifs.append(motif)
        if len(overlap_motifs) > 0:
            os.makedirs(f"overlaps/{pos}", exist_ok=True)
            atlas_motif.to_cif(f"overlaps/{pos}/atlas.cif")
            for motif in overlap_motifs:
                motif.to_cif(f"overlaps/{pos}/ml_{count}.cif")
                count += 1
            pos += 1
    exit()


if __name__ == "__main__":
    cli()
