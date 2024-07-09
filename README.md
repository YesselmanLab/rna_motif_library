# RNA Motif Library

A Python package for RNA Motif Library creation.<br>
preprint: (insert paper link here)

## Data Download

A default CSV (`nrlist_3.320_3.5A.csv`) is in the directory `data/csvs`.<br>
Make sure to download the most recent data (3.5 Å resolution) from:
http://rna.bgsu.edu/rna3dhub/nrlist

In the directory `data/csvs`, delete the default CSV file and replace with your download .<br>

## Installation

```bash
# Create a new conda environment
conda create --name rna_motif_env python=3.8
conda activate rna_motif_env

# Clone the repository and navigate to project directory
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Install the package
pip install .

```

## Creating the library

```bash
# To create the library
python update_library.py
# Everything will run and you will have the results when it's done
```

When finished, you will see several new directories, CSVs, and figures in the project directory. <br>
`motifs` - motifs found in the non-redundant set go here, categorized by type, size, and sequence
`interactions` - individual residues which hydrogen-bond with each other go here, classified by which residues are interacting
`tertiary_contacts` - tertiary contacts found go here, classified by what two types of motifs are in the contact

## Figure generation

The figures used were generated whilst running `update_library.py` using the default CSV inside the directory `data/csvs`.<br>
For further details, check out `figure_plotting.py`.

Figures 2 and 3 are PNGs; they are in the project directory.<br>
Figure 4 also consists of PNGs, however, every interaction/atom combination gets its own figure.<br>
These figures can be found in the directory `heatmaps`.<br>
Data for each respective figure is broken down in CSV files, which are in `heatmap_data`.<br>


