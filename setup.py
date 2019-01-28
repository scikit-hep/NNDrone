import sys
from setuptools import setup
from setuptools import find_packages as findPKGs

pkgs = findPKGs(include=['nndrone*', 'utilities*'])

setup(
   name='nndrone',
   version='1.0',
   description='ML drone creation',
   author='Sean Benson, Konstantin Gizdov',
   author_email='sean.benson@cern.ch, k.gizdov@cern.ch',
   url='https://github.com/scikit-hep/NNDrone',
   packages=pkgs,
   install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'keras']
)
