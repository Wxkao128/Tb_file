# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:33:51 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

# Define constants
e0 = 0
t  = -2.7 

# Define the Hamiltonian function
def ham(kx, ky):
    return e0 + 2*t*(np.cos(kx) + 2*np.cos(kx/2)*np.cos(np.sqrt(3)/2*ky))

# Create a grid of kx and ky values
kx = np.linspace(-2*np.pi, 2*np.pi, 100)    
ky = np.linspace(-2*np.pi, 2*np.pi, 100)
KX, KY = np.meshgrid(kx, ky)

# Calculate the energy values on the grid
EK = ham(KX, KY)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(KX, KY, EK, shading='auto', cmap='GnBu')
plt.colorbar(label='Energy')

# Add contour lines
contours = plt.contour(KX, KY, EK, colors='k', linewidths=0.5, linestyles= 'dashdot' )
# text for contour lines
#plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

# Set labels and title
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Energy Heatmap with Contours')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define constants
t = -2.7

# Define the Hamiltonian function
def ham(kx, ky, e0):
    return e0 + 2*t*(np.cos(kx) + 2*np.cos(kx/2)*np.cos(np.sqrt(3)/2*ky))

# Create a grid of kx and ky values
kx = np.linspace(-2*np.pi, 2*np.pi, 100)
ky = np.linspace(-2*np.pi, 2*np.pi, 100)
KX, KY = np.meshgrid(kx, ky)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize the plot elements
cax = ax.pcolormesh(KX, KY, ham(KX, KY, -5), shading='auto', cmap='GnBu')
contours = ax.contour(KX, KY, ham(KX, KY, -5), colors='k', linewidths=0.5)

# Add colorbar
cbar = fig.colorbar(cax, ax=ax, label='Energy')

# Update function for the animation
def update(frame):
    e0 = frame
    EK = ham(KX, KY, e0)
    for coll in ax.collections:
        coll.remove()  # Remove previous contours and pcolormesh
    cax = ax.pcolormesh(KX, KY, EK, shading='auto', cmap='GnBu')
    contours = ax.contour(KX, KY, EK, colors='k', linewidths=0.5)
    ax.set_title(f'Energy Heatmap with Contours (e0={e0:.2f})')
    return cax, contours

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(-5, 5, 100), interval=100)

ani.save('animation.gif', writer='pillow')

plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Energy Heatmap with Contours')
plt.axis('equal')
plt.show()




