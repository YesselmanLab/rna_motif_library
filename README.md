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

When finished, you will see several new directories, CSVs, and figures in the project directory.<br>

## Figure generation

The figures used were generated whilst running `update_library.py` using the default CSV inside the directory `data/csvs`.<br>
For further details, check out `figure_plotting.py`.

Figures 2 and 3 are PNGs; they are in the project directory.<br>
Figure 4 also consists of PNGs, however, every interaction/atom combination gets its own figure.<br>
These figures can be found in the directory `heatmaps`.<br>
Data for each respective figure is broken down in CSV files, which are in `heatmap_data`.<br>


