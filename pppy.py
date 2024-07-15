from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.visualize import view
import numpy as np

# Load your slab structure (replace 'your_slab_file' with the actual filename)
slab = read('/Users/mac/hyu.cif')

# Create a CO2 molecule and adjust it to have a bent shape
CO2 = Atoms('CO2', positions=[
    [0.0, 0.0, 0.0],      # Carbon atom
    [1.16, 1.0, 0.0],    # Oxygen atom
    [-1.16, 1.0, 0.0]    # Oxygen atom
])

# Set the atom number where the CO2 will be adsorbed
atom_number = 93

# Get the position of the specified atom in the slab
adsorption_site = slab[atom_number].position

# Set the height above the surface where CO2 will be placed
adsorption_height = 2.5  # Adjust this value as needed

# Translate the CO2 molecule to the adsorption site
CO2.translate(adsorption_site + np.array([0, 0, adsorption_height]))

C_O_bond_length =1.16
# Ensure the CO2 molecule is above the surface by checking the z-coordinate
if CO2.positions[:, 2].min() < slab.positions[:, 2].max():
    z_shift = slab.positions[:, 2].max() - CO2.positions[:, 2].min() + 1.0
    CO2.translate([0, 0, z_shift])

#rotate CO2 molecule to ensure O atoms are bent upwards
angle=90.0
CO2.rotate(angle, v='x', center='COM')
# Add the CO2 molecule to the slab
slab += CO2

# View the final structure
view(slab)