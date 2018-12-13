from setuptools import setup
try:
    from setuptools import find_namespace_packages as findpacks
except ImportError as e:
    from setuptools import find_packages as findpacks

setup(
   name='nndrone',
   version='1.0',
   description='ML drone creation',
   author='Sean Benson, Konstantin Gizdov',
   author_email='sean.benson@cern.ch, k.gizdov@cern.ch',
   url='https://github.com/scikit-hep/NNDrone',
   packages=findpacks(include=['nndrone*', 'utilities*']),
   install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'keras']
)
