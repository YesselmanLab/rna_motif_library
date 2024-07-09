# RNA Motif Library

A Python package for RNA Motif Library creation.<br>
preprint: (insert paper link here)

## Data Download
Make sure to download the most recent data (3.5 Å resolution) used in this project from:
http://rna.bgsu.edu/rna3dhub/nrlist

Move this CSV file to the directory "data/csvs/" in the current project directory.<br>
An old CSV is in there by default; make sure to delete the old one and use the most recent.

## Installation

```bash
# Clone the repository and navigate to project directory
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install .

```

## Creating the library

```bash
# To create the library
python update_library.py
# Everything will run and you will have the results when it's done
```

## Figure generation

The figures used were generated 





How to run unit tests:<br>
python test_dssr.py - tests DSSR functions<br>
python test_snap.py - tests SNAP functions<br>

