import sys
from setuptools import setup, find_packages

pkgs = None

if sys.version_info <= (3, 3):
    pkgs = find_namespace_packages(include=['nndrone*', 'utilities*'])
else:
    pkgs = find_packages(include=['nndrone*', 'utilities*'])

setup(
   name='nndrone',
   version='1.0',
   description='ML drone creation',
   author='Sean Benson, Konstantin Gizdov',
   packages=pkgs,
   install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'keras']
)
