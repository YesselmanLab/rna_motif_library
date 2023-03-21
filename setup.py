from setuptools import setup

setup(
        # metadata
        name='rna_motif_library',
        version='1.0.0',
        description='a minimal package for rna_motif_library creation',
        long_description_content_type="test/markdown",
        author='Joe Yesselman',
        author_email='jyesselm@unl.edu',
        url='https://github.com/jyesselm/rna_motif_library',
        # package contents
        packages=[
            'rna_motif_library',
        ],
        package_dir={'rna_motif_library': 'rna_motif_library'},
        py_modules=[
            'rna_motif_library/dssr_lib',
            'rna_motif_library/settings',
            'rna_motif_library/snap',
            'rna_motif_library/update_library',
            'rna_motif_library/update_library_nmr',
        ],
        include_package_data=True,
        install_requires=[
            'pandas',
        ],
        zip_safe=False,
        keywords='rna_motif_library',
        classifiers=[
            'Intended Audience :: Developers',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: Implementation :: PyPy',
        ],
        entry_points={
            'console_scripts': [
            ]
        }
)
