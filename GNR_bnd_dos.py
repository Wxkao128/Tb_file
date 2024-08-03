

import numpy as np
import matplotlib.pyplot as plt

# Clearing previous figures if any
plt.close('all')

t = 1.03  # in eV (Electron Volts)
a = 3.82  # in Angstroms
onsite = 0
ka = np.linspace(-np.pi, np.pi, 301)


# Zig Zag Graphene N = 4
A = np.zeros((8, 8),dtype=complex)
for i in range(7):
    A[i, i+1] = t
    A[i+1, i] = t
    
    if i%2==0:
        A[i,i] = onsite
    else:
        A[i,i] = 0

B = np.zeros((8, 8),dtype=complex)
B[0,1] = t
B[3,2] = t
B[4,5] = t
B[7,6] = t


eigE1 = np.zeros((len(ka), 8), dtype=complex)

for idx, k_val in enumerate(ka):
    # Computing eigenvalues and eigenvectors
    V1, D1 = np.linalg.eigh(B.T.conj()*np.exp(-1j*k_val) + A + B*np.exp(1j*k_val))
    eigE1[idx, :] = V1

# Plotting energy vs ka
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
for i in range(eigE1.shape[1]):
    ax1.plot(ka,  eigE1.T[i], c='b')

ax1.set_xlabel('ka')
ax1.set_ylabel('Energy in eV')
ax1.set_title('Band structure for N = 4 Z-SNR')
ax1.grid(True)
ax1.set_xticks([-np.pi, 0, np.pi])
ax1.set_xticklabels(['-π', '0', 'π'])
ax1.axvline(x=np.pi, color='black')
ax1.axvline(x=-np.pi, color='black')
ax1.set_xlim([-np.pi, np.pi])
ax1.set_ylim([-4, 4])


# DOS
eu = 4
el = -4
Ne = 100
delta = 1e-2

Fermi_energy_array = np.linspace(el, eu, 301)  # 计算中取的费米能Fermi_energy组成的数组
#dos_fn = np.zeros(eigE1.T[0].shape, dtype=complex)

dos_fn = np.zeros(Fermi_energy_array.shape[0]) 
for i in range(Fermi_energy_array.shape[0]):
    
    delta_value = 0 #for sum all over k
    for bnd_index in range(eigE1.T.shape[0]):
        for k in range(len(ka)):
            delta_value += 1/(np.pi) * (delta/((Fermi_energy_array[i]-eigE1.T[bnd_index][k])**2+delta**2))
    
    dos_fn[i] = delta_value.real

ax2.plot(dos_fn, Fermi_energy_array)
ax2.set_title('Density of state')
ax2.set_ylabel('E(eV)')
ax2.set_xlabel('DOS')
ax2.set_ylim([-4, 4])
ax2.grid(True)

# 调整布局
plt.tight_layout()
plt.show()



