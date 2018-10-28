from setuptools import setup, find_packages

setup(
   name='NNDrone',
   version='1.0',
   description='ML drone creation',
   author='Sean Benson, Konstantin Gizdov',
   packages=find_packages(),
   install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'keras']
)
