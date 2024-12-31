import pandas as pd
import os
import json
from typing import Dict

from rna_motif_library.classes import get_residues_from_pdb

RESOURCES_PATH = os.path.join("test", "resources")


def test_residues_from_json():
    residues = get_residues_from_pdb(os.path.join(RESOURCES_PATH, "HELIX.IDEAL.pdb"))
    assert len(residues) == 4
