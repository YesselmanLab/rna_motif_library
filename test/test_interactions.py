import json
import os

from rna_motif_library.classes import Hbond, Basepair
from rna_motif_library.interactions import get_hbonds_and_basepairs


def test_json_hbonds():
    hbonds, _ = get_hbonds_and_basepairs("1MDG")
    json_path = os.path.join("test", "resources", "hbonds.json")
    with open(json_path, "w") as f:
        json.dump([hbond.to_dict() for hbond in hbonds], f)
    data = json.load(open(json_path))
    json_hbonds = [Hbond.from_dict(hbond) for hbond in data]
    assert len(hbonds) == len(json_hbonds)
    for hbond, json_hbond in zip(hbonds, json_hbonds):
        assert hbond == json_hbond


def test_json_basepairs():
    _, basepairs = get_hbonds_and_basepairs("1MDG")
    json_path = os.path.join("test", "resources", "basepairs.json")
    with open(json_path, "w") as f:
        json.dump([bp.to_dict() for bp in basepairs], f)
    data = json.load(open(json_path))
    json_basepairs = [Basepair.from_dict(bp) for bp in data]
    assert len(basepairs) == len(json_basepairs)
    for bp, json_bp in zip(basepairs, json_basepairs):
        assert bp == json_bp
