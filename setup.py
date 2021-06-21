from setuptools import setup, find_packages
from glob import glob
__version__ = "1.0.0"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyDNTNK',
    version=__version__,
    author='Manish Bhattarai, Erik Skau, Phan Minh Duc Truong, Maksim E. Eren, Namita Kharat, Raviteja Vangara, Hristo Djidjev, Boian Alexandrov',
    author_email='ceodspspectrum@lanl.gov',
    description='Python Distributed Non Negative Matrix Factorization with determination of hidden features',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'pyDNTNK': 'pyDNTNK/'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    dependency_links=['https://github.com/lanl/pyDNMFk/tarball/main#egg=pyDNMFk-1.0'],
    packages=find_packages(),
    setup_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'mpi4py', 'pytest-mpi','scikit-learn', 'pytest', 'zarr', 'dask'],
    python_requires='>=3.8.5',
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
    license='License :: BSD3 License',
)
