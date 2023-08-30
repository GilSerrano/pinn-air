from setuptools import setup

setup(
   name='pinn_air',
   version='1.0',
   description='Package that implements the PINNAir model',
   author='Gil Serrrano, Marcelo Jacinto, Jose Gomes and Joao Pinto',
   author_email='gil.serrano@tecnico.ulisboa.pt, marcelo.jacinto@tecnico.ulisboa.pt, josepgomes@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt',
   packages=['pinn_air'],  #same as name
   install_requires=['numpy', 'matplotlib', 'pytorch'], #external packages as dependencies
)