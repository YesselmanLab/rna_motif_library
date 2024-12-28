from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="rna_motif_library",
    version="0.2.0",
    description="A minimal package for RNA motif library creation",
    long_description_content_type="text/markdown",
    author="Joe Yesselman",
    author_email="jyesselm@unl.edu",
    url="https://github.com/jyesselm/rna_motif_library",
    packages=[
        "rna_motif_library",
    ],  # Automatically find packages
    py_modules=[
        "rna_motif_library/dssr",
        "rna_motif_library/dssr_hbonds",
        "rna_motif_library/settings",
        "rna_motif_library/snap",
        "rna_motif_library/tert_contacts",
        "rna_motif_library/update_library",
        "rna_motif_library/classes",
        "rna_motif_library/cli",
        "rna_motif_library/logger",
        "rna_motif_library/tranforms",
        "rna_motif_library/util",
    ],
    include_package_data=True,
    # install_requires=requirements,
    zip_safe=False,
    keywords="RNA motif library",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={},
)
