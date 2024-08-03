# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:40:17 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

#=============================================================================

#=============================================================================

filer1 = 'POSCAR_TBG_2179_supercell.vasp'
#lattice_vector=np.loadtxt(filer1,skiprows=2,max_rows=3)
atoms=np.genfromtxt(filer1,skip_header=8,dtype=float)
print('atom coordinates:')
print(atoms.shape)

# Vertex coordinates of supercell
ext = [(3.25363,  16.90635,  13.40000),
       (3.25363,   5.63545,  13.40000),
       (9.76089,   5.63545,  13.40000),
       (9.76089,  16.90635,  13.40000),
       (3.25363,  16.90635,  13.40000)]


# Specify the boundary coordinates of the square
x_min, x_max = 3.25363, 9.76089
y_min, y_max = 5.63545, 16.90635
z_min, z_max = 10.0000, 13.40000
L1 = [x_max-x_min,0,0]
L2 = [0,y_max-y_min,0]
L3 = [0,0,20]

# Check if each point is within the boundary of the square
x, y, z = np.transpose(atoms)  # Obtain the coordinates of x, y, and z directions
inside = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max) & (z_min <= z) & (z <= z_max)
# Convert the judgment results into a Boolean array

# Output the points that are inside the boundary of the square
print(atoms[inside])
print(atoms[inside].shape)
plt.scatter(atoms[inside][:,0],atoms[inside][:,1])
plt.show()

# Write the coordinates of the points inside a square boundary to a txt file
filename = "POSCAR_supercell"
with open(filename, "w") as f:
    f.write('Tiwsted bilayer graphene ribbon\n')
    f.write('1.0\n') #scaling factor
    for item in L1:
        f.write("%s " % item)
    f.write('\n')
    for item in L2:
        f.write("%s " % item)
    f.write('\n')
    for item in L3:
        f.write("%s " % item)
    f.write('\n')
    f.write('C\n') #elements 
    f.write("%s\n" %(atoms[inside].shape[0]))
    f.write('C\n') #Cartesian
    
    # Store the coordinates to be written in a string first
    data = "\n".join([f"{atoms[i, 0]} {atoms[i, 1]} {atoms[i, 2]}" for i in range(len(atoms)) if inside[i]])
    # Write to the file at once
    f.write(data)
    