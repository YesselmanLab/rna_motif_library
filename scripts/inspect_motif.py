import click
from typing import List, Optional
from rna_motif_library.basepair import get_cached_basepairs
from rna_motif_library.chain import get_cached_chains, Chains, write_chain_to_cif
from rna_motif_library.motif import get_cached_motifs, Motif
from rna_motif_library.residue import Residue
from rna_motif_library.logger import setup_logging
from rna_motif_library.util import wc_basepairs_w_gu
from rna_motif_library.hbond import get_cached_hbonds
from rna_motif_library.x3dna import get_cached_dssr_output, X3DNAResidueFactory
from rna_motif_library.motif_factory import MotifFactory

setup_logging()


def setup_motif_factory(pdb_id: str) -> MotifFactory:
    """Create and return a MotifFactory instance for the given PDB ID.

    Args:
        pdb_id: PDB identifier

    Returns:
        Configured MotifFactory instance
    """
    basepairs = get_cached_basepairs(pdb_id)
    chains = get_cached_chains(pdb_id)
    hbonds = get_cached_hbonds(pdb_id)
    rna_chains = Chains(chains)
    return MotifFactory(pdb_id, rna_chains, basepairs, hbonds)


def filter_motifs(
    motifs: List[Motif],
    motif_type: str,
    motif_min_size: int,
    motif_max_size: int,
):
    """Filter motifs based on type, size, and PDB ID"""
    if motif_type is not None:
        motifs = [motif for motif in motifs if motif.mtype == motif_type]
    if motif_min_size is not None:
        motifs = [motif for motif in motifs if motif.num_residues() >= motif_min_size]
    if motif_max_size is not None:
        motifs = [motif for motif in motifs if motif.num_residues() <= motif_max_size]
    return motifs


def get_motifs_with_residue(motifs: List[Motif], residue: Residue) -> List[Motif]:
    matching_motifs = []
    for motif in motifs:
        if residue in motif.get_residues():
            matching_motifs.append(motif)
    return matching_motifs


def motif_filter_options(f):
    """Common options for filtering motifs"""
    f = click.option("-mt", "--motif-type", default=None, help="Filter by motif type")(
        f
    )
    f = click.option(
        "-ms",
        "--motif-min-size",
        default=None,
        type=int,
        help="Minimum motif size in nucleotides",
    )(f)
    f = click.option(
        "-mx",
        "--motif-max-size",
        default=None,
        type=int,
        help="Maximum motif size in nucleotides",
    )(f)
    return f


@click.group()
def cli():
    """Tools for inspecting RNA motifs"""
    pass


@cli.command()
@click.argument("motif_name")
def inspect(motif_name: str):
    """Inspect details of a specific motif"""
    spl = motif_name.split("-")
    pdb_id = spl[-2]
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {motif.name: motif for motif in motifs}
    motif = motifs_by_name[motif_name]

    motif_factory = setup_motif_factory(pdb_id)
    motif_factory.inspect_motif(motif, motifs)


@cli.command()
@click.argument("pdb_id")
@motif_filter_options
@click.option("--pdbs", is_flag=True)
def list_motifs(
    pdb_id: str, motif_type: str, motif_min_size: int, motif_max_size: int, pdbs: bool
):
    """List all motifs in a PDB structure"""
    motifs = get_cached_motifs(pdb_id)
    motifs = filter_motifs(motifs, motif_type, motif_min_size, motif_max_size)
    for motif in motifs:
        print(f"{motif.name}: {motif.mtype} ({motif.num_residues()} nt)")
        if pdbs:
            motif.to_cif()


@cli.command()
@click.argument("motif_name")
def neighbors(motif_name: str):
    """Show neighboring motifs"""
    spl = motif_name.split("-")
    pdb_id = spl[-2]
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {motif.name: motif for motif in motifs}
    motif = motifs_by_name[motif_name]
    hbonds = get_cached_hbonds(pdb_id)
    basepairs = get_cached_basepairs(pdb_id)
    chains = get_cached_chains(pdb_id)
    rna_chains = Chains(chains)
    motif_factory = MotifFactory(pdb_id, rna_chains, basepairs, hbonds)
    motif_factory.inspect_motif_neighbors(motif, motifs)


@cli.command()
@click.argument("pdb_id")
@motif_filter_options
def interactions(
    pdb_id: str, motif_type: str, motif_min_size: int, motif_max_size: int
):
    """Show interactions between motifs"""
    motifs = get_cached_motifs(pdb_id)
    motifs = filter_motifs(motifs, motif_type, motif_min_size, motif_max_size)

    motif_factory = setup_motif_factory(pdb_id)
    interactions = motif_factory.find_motif_interactions(motifs)
    for motif1, motif2, num_shared_bps in interactions:
        print(f"{motif1.name} and {motif2.name} share {num_shared_bps} basepairs")
        motif1.to_cif()
        motif2.to_cif()


@cli.command()
@click.argument("motif_name")
def test(motif_name: str):
    basepairs = get_cached_basepairs("8C3A")
    """
    dssr_output = get_cached_dssr_output("8C3A")
    pairs = dssr_output.get_pairs()
    for pair in pairs.values():
        res_1_num = pair.nt1.nt_resnum
        res_2_num = pair.nt2.nt_resnum
        if res_1_num == 1930 and res_2_num == 1914:
            print(pair.hbonds_desc)
        elif res_1_num == 1914 and res_2_num == 1930:
            print(pair.hbonds_desc)
    """

    motifs = get_cached_motifs("8C3A")
    motifs_by_name = {motif.name: motif for motif in motifs}
    spl = motif_name.split("-")
    pdb_id = spl[-2]
    mf = setup_motif_factory(pdb_id)
    possible_hairpins = mf.get_looped_strands()
    helices = mf.get_helices(possible_hairpins)
    m = motifs_by_name[motif_name]
    residues = m.get_residues()
    residues_by_name = {r.get_x3dna_str(): r for r in residues}
    for h in possible_hairpins:
        bps = mf._get_basepairs_for_strands(h.strands)
        bp_count = 0
        # for bp in bps:
        #    if bp.bp_type in wc_basepairs_w_gu and bp.lw == "cWW":
        #        bp_count += 1
        print(h.name, len(h.strands), bp_count)
        if bp_count > 1:
            print(h.name)
            h.to_cif()

    exit()


@cli.command()
@click.argument("motif_name")
def shared_bp_motifs(motif_name: str):
    spl = motif_name.split("-")
    pdb_id = spl[-2]
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {motif.name: motif for motif in motifs}
    m = motifs_by_name[motif_name]
    m.to_cif()
    mf = setup_motif_factory(pdb_id)
    bp_ends = []
    for bp in m.basepair_ends:
        bp_ends.append(bp.res_1.get_str() + "-" + bp.res_2.get_str())
        bp_ends.append(bp.res_2.get_str() + "-" + bp.res_1.get_str())
    for m in motifs:
        for bp in m.basepair_ends:
            if bp.res_1.get_str() + "-" + bp.res_2.get_str() in bp_ends:
                print(m.name)
                m.to_cif()


if __name__ == "__main__":
    cli()
