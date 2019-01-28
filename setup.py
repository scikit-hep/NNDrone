import sys
from setuptools import setup
from setuptools import find_packages as findPKGs

pkgs = None
pkgs = findPKGs(include=['nndrone*', 'utilities*'])

setup(
   name='nndrone',
   version='1.0',
   description='ML drone creation',
   author='Sean Benson, Konstantin Gizdov',
   packages=pkgs,
   install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'keras']
)
