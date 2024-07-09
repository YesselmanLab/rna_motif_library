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
# Clone the repository
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Set up virtual environment
python -m venv venv
source venv/bin/activate


```

To run the code:<br>
python update_library.py - will create the library






How to run unit tests:<br>
python test_dssr.py - tests DSSR functions<br>
python test_snap.py - tests SNAP functions<br>

