from setuptools import setup, find_packages

setup(
    name='rna_motif_library',
    version='1.0.0',
    description='A minimal package for RNA motif library creation',
    long_description_content_type="text/markdown",
    author='Joe Yesselman',
    author_email='jyesselm@unl.edu',
    url='https://github.com/jyesselm/rna_motif_library',
    packages=find_packages(),  # Automatically find packages
    py_modules=[
        'dssr',
        'dssr_hbonds',
        'figure_plotting',
        'settings',
        'snap',
        'tertiary_contacts',
        'update_library'
    ],
    include_package_data=True,
    install_requires=[
        'biopandas~=0.4.1',
        'pandas~=1.5.3',
        'wget~=3.2',
        'setuptools~=60.2.0',
        'requests~=2.28.2',
        'numpy~=1.24.2',
        'seaborn~=0.13.1',
        'matplotlib~=3.7.1',
        'pydssr @ git+https://github.com/YesselmanLab/py_dssr.git#egg=pydssr',
        'click~=8.1.7',
        'tqdm~=4.66.4'
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
