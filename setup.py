from setuptools import setup

setup(
    name='rna_motif_library',
    version='1.0.0',
    description='A minimal package for RNA motif library creation',
    long_description_content_type="text/markdown",
    author='Joe Yesselman',
    author_email='jyesselm@unl.edu',
    url='https://github.com/jyesselm/rna_motif_library',
    packages=['rna_motif_library'],
    package_dir={'rna_motif_library': 'rna_motif_library'},
    py_modules=[
        'rna_motif_library.dssr',
        'rna_motif_library.dssr_hbonds',
        'rna_motif_library.figure_plotting',
        'rna_motif_library.settings',
        'rna_motif_library.snap',
        'rna_motif_library.tertiary_contacts',
        'rna_motif_library.update_library',
        'rna_motif_library.update_library_nmr'
    ],
    include_package_data=True,
    install_requires=[
        'pydssr~=0.0.3',
        'biopandas~=0.4.1',
        'pandas~=1.5.3',
        'wget~=3.2',
        'setuptools~=60.2.0',
        'requests~=2.28.2',
        'numpy~=1.24.2',
        'seaborn~=0.13.1',
        'matplotlib~=3.7.1'
    ],
    zip_safe=False,
    keywords='RNA motif library',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    entry_points={}
)
