# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:53:00 2023

@author: User
"""
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

h_bar = 1.054572 * 10**(-27)
vf = 1 * 10**8
d0 = 1 * 10**(-7)
eV = 1.6021766208 * 10**(-12)
k = eV * d0 / h_bar / vf

def T(x, E0, V, d):
    return ((E0 - V)**2 - E0**2 * np.sin(x)**2) / ((E0 - V)**2 - E0**2 * np.sin(x)**2 + V**2 * np.sin(k * d * np.sqrt((E0 - V)**2 - E0**2 * np.sin(x)**2))**2 * np.tan(x)**2)

E0_values = np.linspace(-0.1, 0.5, 1000)
x_values = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)

X, Y = np.meshgrid(E0_values, x_values)
Z = T(X, Y, 0.2, 30)

plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[-0.1, 1.5, -np.pi/2 + 0.01, np.pi/2 - 0.01], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='T')
plt.xlabel('E, eV')
plt.ylabel('θ')
plt.title('T')
plt.show()

def G(E0, V, d):
    integrand = lambda x: np.cos(x) * T(x, E0, V, d)
    return quad(integrand, -np.pi/2, np.pi/2)[0]

E0_values = np.arange(-1, 1.5, 0.001)
lst = [[E0, G(E0, 0.2, 30)] for E0 in E0_values]

lst = np.array(lst)
plt.figure(figsize=(8, 6))
plt.plot(lst[:, 0], lst[:, 1])
plt.xlabel('E, eV')
plt.ylabel('G')
plt.show()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

h_bar = 1.054572 * 10**-27
vf = 1 * 10**8
d0 = 1 * 10**-7
eV = 1.6021766208 * 10**-12
k = eV * d0 / h_bar / vf

def T(x, E0, V, d):
    return ((E0 - V)**2 - E0**2 * np.sin(x)**2) / \
        ((E0 - V)**2 - E0**2 * np.sin(x)**2 + V**2 * np.sin(k * d * np.sqrt((E0 - V)**2 - E0**2 * np.sin(x)**2))**2 * np.tan(x)**2)

def G(E0, V, d):
    return quad(lambda x: np.cos(x) * T(x, E0, V, d), -np.pi / 2, np.pi / 2)[0]

E0_range = np.arange(-1, 1.501, 0.001)
lst = [[E0, G(E0, 0.2, 30)] for E0 in E0_range]

plt.figure()
plt.plot(*zip(*lst))
plt.xlabel('E, eV')
plt.ylabel('G')
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

h_bar = 1.054572 * 10**-27
vf = 10**8
d0 = 10**-7
eV = 1.6021766208 * 10**-12
k = eV * d0 / h_bar / vf

def T(x, E0, V, d):
    return ((E0 - V)**2 - E0**2 * np.sin(x)**2) / ((E0 - V)**2 - 
      E0**2 * np.sin(x)**2 + V**2 * np.sin(k * d * np.sqrt((E0 - V)**2 - E0**2 * np.sin(x)**2))**2 * np.tan(x)**2)

def G(E0, V, d):
    return quad(lambda x: np.cos(x) * T(x, E0, V, d), -np.pi/2, np.pi/2)[0]

E0_values = np.arange(-1, 1.5, 0.001)
lst = [[E0, G(E0, 0.2, 30)] for E0 in E0_values]

plt.figure(figsize=(8, 6))
plt.plot([item[0] for item in lst], [item[1] for item in lst])
plt.xlabel('E, eV')
plt.ylabel('G')
plt.xlim(-1,1.5)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# Define the range for E0 and x
E0 = np.linspace(-0.1, 0.5, 100)
x = np.linspace(-np.pi/2 + 10**-2, np.pi/2 - 10**-2, 100)

# Create a meshgrid for the E0 and x values
E0, x = np.meshgrid(E0, x)

# Compute the values of T for the meshgrid
Z = T(x, E0, 0.2, 30)

# Create the density plot
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[-0.1, 0.5, -np.pi/2 + 10**-2, np.pi/2 - 10**-2], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='T')
plt.xlabel('E, eV')
plt.ylabel('θ')
plt.title('T')
plt.show()
