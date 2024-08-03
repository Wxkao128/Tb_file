#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:50:01 2023

@author: wxkao
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.linalg import kron
#from scipy.linalg import eigvalsh
#from scipy.linalg import eig
import math
import time


d = 1.42    # Å, nearest c-c bond length
vf = 5.944  # eV·Å Fermi velocity
phi = 2 * np.pi / 3
angle = 1.05 # degree
theta = angle / 180 * np.pi # degree to arc for model
w1 = 0.110
Tb  = w1 * np.ones((2, 2),dtype=complex)
Ttr = w1 * np.array([[np.exp(-1j*phi), 1], [np.exp( 1j*phi), np.exp(-1j*phi)]],dtype=complex)
Ttl = w1 * np.array([[np.exp( 1j*phi), 1], [np.exp(-1j*phi), np.exp( 1j*phi)]],dtype=complex)
qb  = 8 * np.pi * np.sin(theta/2) / (3*np.sqrt(3)*d) * np.array([0, -1])
qtr = 8 * np.pi * np.sin(theta/2) / (3*np.sqrt(3)*d) * np.array([ np.sqrt(3)/2, 1/2])
qtl = 8 * np.pi * np.sin(theta/2) / (3*np.sqrt(3)*d) * np.array([-np.sqrt(3)/2, 1/2])
tr = 2  # truncation
s = (2*tr+1)
N = s**2
H = np.zeros((2*2*N, 2*2*N),dtype=complex)  # Hamiltonian

nkp = 50 #number of k points in k path section
N1y = np.linspace(-1, 0, nkp, endpoint=False) * np.linalg.norm(qb)
N1x = 0 * N1y
N2y = np.linspace( 0, 1, nkp, endpoint=False) * np.linalg.norm(qb)
N2x = 0 * N2y
N3y = np.linspace( 1, -1/2, math.ceil(nkp*np.sqrt(3)), endpoint=False) * np.linalg.norm(qb)
N3x = (-N3y + np.linalg.norm(qb)) / np.sqrt(3)
N4y = np.linspace(-1/2, -1, nkp) * np.linalg.norm(qb)
N4x = (N4y + np.linalg.norm(qb)) * np.sqrt(3)
Nx = np.concatenate((N1x, N2x, N3x, N4x), axis=None)
Ny = np.concatenate((N1y, N2y, N3y, N4y), axis=None)  # k-path:A-B-C-D-A
band = np.zeros((len(Nx), 4 * N))
print('parameters are set')


def HSLG(vf, qx, qy, kk, jj, d, theta):
    b1m = 8 * np.pi * np.sin(np.abs(theta)) / (3*d) * np.array([1/2, -np.sqrt(3)/2])
    b2m = 8 * np.pi * np.sin(np.abs(theta)) / (3*d) * np.array([1/2,  np.sqrt(3)/2])
    kx = qx - kk * b1m[0] - jj * b2m[0]
    ky = qy - kk * b1m[1] - jj * b2m[1]
    h = -vf * np.sqrt(kx**2 + ky**2) * np.array([[0, np.exp( 1j * (np.angle(kx+1j*ky) - theta))],
                                                    [np.exp(-1j * (np.angle(kx+1j*ky) - theta)), 0]])
    return h

t1 = time.time()
#'''
for ii in range(len(Nx)):
    t1_1 = time.time()
    kx = Nx[ii]
    ky = Ny[ii]
    c = 1
    Hdiag = np.zeros((2*2*N, 2*2*N), dtype=complex)  # diagonal element
    for kk in range(-tr, tr+1):
        for jj in range(-tr, tr+1):
            temp = np.zeros((2*N, 2*N),dtype=complex)
            temp[c-1, c-1] = 1
            Hdiag = Hdiag + np.kron(temp, HSLG(vf, kx, ky, kk, jj, d, theta/2))  # 1st layer, red points
            c += 1

    for kk in range(-tr, tr+1):
        for jj in range(-tr, tr+1):
            temp = np.zeros((2*N, 2*N),dtype=complex)
            temp[c-1, c-1] = 1
            Hdiag = Hdiag + np.kron(temp, HSLG(vf, kx - qb[0], ky - qb[1], kk, jj, d, -theta/2))  # 2nd layer, blue points
            c += 1

        
    Hoff1 = np.zeros((2*N, 2*N),dtype=complex)  # off-diagonal element
    Hoff2 = np.zeros((2*N, 2*N),dtype=complex)
   
    for k2 in range(-tr, tr+1):
        for j2 in range(-tr, tr+1):
            for k1 in range(-tr, tr+1):
                for j1 in range(-tr, tr+1):
                    if (k1 == k2) and (j1 == j2):
                        off1 = np.zeros((N, N),dtype=complex)
                        off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Tb)
                        
                        off2 = np.zeros((N, N),dtype=complex)
                        off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Tb.T))
                        
                    elif (k1 == k2) and (j1 + 1 == j2):
                        off1 = np.zeros((N, N),dtype=complex)
                        off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Ttr)
                        
                        off2 = np.zeros((N, N),dtype=complex)
                        off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttr.T))
                        
                    elif (k1 - 1 == k2) and (j1 == j2):
                        off1 = np.zeros((N, N),dtype=complex)
                        off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Ttl)
                        
                        off2 = np.zeros((N, N),dtype=complex)
                        off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttl.T))

    Hoff = np.kron(np.array([[0, 1], [0, 0]]), Hoff1) + np.kron(np.array([[0, 0], [1, 0]]), Hoff2)
    H = Hdiag + Hoff
    
    # use scipy for solving eigenvalue problems
    # use scipy.linalg.eig
    #EE = eig(H)
    #band[ii, :] = np.sort(EE[0])
    
    # use numpy to solve eigenvalue problems
    EE = np.linalg.eigh(H)
    band[ii, :] = np.sort(EE[0])
    
    t1_2 = time.time()
    #print(f'loop {ii:3} cost time: {(t1_2-t1_1):.2f}')

t2 = time.time()


#tick_coord = [0, len(N1x), len(N1x)+len(N2x),len(N1x)+len(N2x)+math.ceil(len(N3x)*0.545),  
#              len(N1x)+len(N2x)+len(N3x),len(N1x)+len(N2x)+len(N3x)+len(N4x)]
#tick_label = ["K","K'",'G','M','G','K']

tick_coord = [0, len(N1x), len(N1x)+len(N2x),len(N1x)+len(N2x)+len(N3x),
              len(N1x)+len(N2x)+len(N3x)+len(N4x)]

fig = plt.figure()
for ii in range(len(band[1])):
    plt.plot(np.arange(0, len(Nx)), band[:, ii], '-k')
    #plt.axis([0, len(Nx)-1, -0.8, 0.8])
    
for i in tick_coord:    
    plt.axvline(x=i,ls=':')
    
plt.ylim(-0.25,0.25)
#plt.ylim(-1.,1.)
#plt.xticks(tick_coord,tick_label)    
plt.xlim(-2,len(Nx)+2)
plt.show()

t3 = time.time()
print(f'calculate cost time: {(t2-t1):.2f}')
print(f'    total cost time: {(t3-t1):.2f}')
#'''

# 2d band
'''

t1_1 = time.time()
nkp = 30 #number of k points in k path section
kxr = 0.6 #kx range
kyr = 0.6 #ky range
Nxx = np.linspace(-kxr,kxr,nkp)
Nyy = np.linspace(-kyr,kyr,nkp)
band = np.zeros((len(Nxx),len(Nyy), 4 * N))
#Nx = np.concatenate((N1x, N2x, N3x, N4x), axis=None)
#Ny = np.concatenate((N1y, N2y, N3y, N4y), axis=None)  # k-path:A-B-C-D-A
#band = np.zeros((len(Nx), 4 * N))
print('parameters are set')

del Nx,Ny

def ham(Nx,Ny):
    for sy in range(len(Ny)):
        for sx in range(len(Nx)):
            #t1_1 = time.time()
            kx = Nx[sx]
            ky = Ny[sy]
            c = 1
            Hdiag = np.zeros((2*2*N, 2*2*N), dtype=complex)  # diagonal element
            for kk in range(-tr, tr+1):
                for jj in range(-tr, tr+1):
                    temp = np.zeros((2*N, 2*N),dtype=complex)
                    temp[c-1, c-1] = 1
                    Hdiag = Hdiag + np.kron(temp, HSLG(vf, kx, ky, kk, jj, d, theta/2))  # 1st layer, red points
                    c += 1
        
            for kk in range(-tr, tr+1):
                for jj in range(-tr, tr+1):
                    temp = np.zeros((2*N, 2*N),dtype=complex)
                    temp[c-1, c-1] = 1
                    Hdiag = Hdiag + np.kron(temp, HSLG(vf, kx - qb[0], ky - qb[1], kk, jj, d, -theta/2))  # 2nd layer, blue points
                    c += 1
        
                
            Hoff1 = np.zeros((2*N, 2*N),dtype=complex)  # off-diagonal element
            Hoff2 = np.zeros((2*N, 2*N),dtype=complex)
           
            for k2 in range(-tr, tr+1):
                for j2 in range(-tr, tr+1):
                    for k1 in range(-tr, tr+1):
                        for j1 in range(-tr, tr+1):
                            if (k1 == k2) and (j1 == j2):
                                off1 = np.zeros((N, N),dtype=complex)
                                off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                                Hoff1 = Hoff1 + np.kron(off1, Tb)
                                
                                off2 = np.zeros((N, N),dtype=complex)
                                off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                                Hoff2 = Hoff2 + np.kron(off2, np.conj(Tb.T))
                                
                            elif (k1 == k2) and (j1 + 1 == j2):
                                off1 = np.zeros((N, N),dtype=complex)
                                off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                                Hoff1 = Hoff1 + np.kron(off1, Ttr)
                                
                                off2 = np.zeros((N, N),dtype=complex)
                                off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                                Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttr.T))
                                
                            elif (k1 - 1 == k2) and (j1 == j2):
                                off1 = np.zeros((N, N),dtype=complex)
                                off1[(k1+tr)*s + j1 + tr, (k2+tr)*s + j2 + tr] = 1
                                Hoff1 = Hoff1 + np.kron(off1, Ttl)
                                
                                off2 = np.zeros((N, N),dtype=complex)
                                off2[(k2+tr)*s + j2 + tr, (k1+tr)*s + j1 + tr] = 1
                                Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttl.T))
        
            Hoff = np.kron(np.array([[0, 1], [0, 0]]), Hoff1) + np.kron(np.array([[0, 0], [1, 0]]), Hoff2)
            H = Hdiag + Hoff
            
            # use scipy for solving eigenvalue problems
            # use scipy.linalg.eig
            #EE = eig(H)
            #band[ii, :] = np.sort(EE[0])
            
            # use numpy to solve eigenvalue problems
            EE = np.linalg.eigh(H)
            #band[ii, :] = np.sort(EE[0])
            
            band[sx,sy,:] = np.sort(EE[0])

            t1_2 = time.time()
            #print(f'loop {ii:3} cost time: {(t1_2-t1_1):.2f}')
    return band


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(Nxx, Nyy)

for i in range(2*N-5,2*N+5):
    Z = ham(Nxx,Nyy)[:,:,i]
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-0.25,0.25)

t1_2 = time.time()
print(f'cost time: {(t1_2-t1_1):.2f}')
plt.show()

'''
