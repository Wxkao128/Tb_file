#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:18:04 2024

@author: wxkao
"""

import numpy as np
import matplotlib.pyplot as plt
import time
#import sys
from scipy.spatial import distance
#from scipy.sparse.linalg import eigsh
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#import cupy as cp

###############################################################################
###
### Preparation and defining some variables
###
###############################################################################
# now: 22 23 should be plot. kdens=7 ksection=5
time_1 = time.time() 
c = 3**(1/2)
d = 0.142  #nm bond length
a = 0.246  #nm lattice constant(算supercell的晶格常數才用到)
h = 0.335  #nm interlayer distance for bilayer graphene
h1= 1      #nm, bottom layer height, for 'POSCAR' and 'count atom number'
h2= h1+h   #nm,    top layer height
n1 = 1 
m1 = 2   
kdens = 15 
nm1 = (m1**2+n1**2+4*m1*n1)/(2*(m1**2+n1**2+m1*n1))
t1 = np.arccos(nm1)        #twisted angle (arc)
angle1 = round((t1/np.pi)*180,2)  #twisted angle (degree)
print('twisted angle: %s°'%(angle1))

stack_conf = 'AA' # 'AA' or 'AB' stacking
plotb = 1  # determine wether plot band or not directly
###############################################################################
"""
Formula: 
    
    1    m^2 + n^2 + 4mn
  ----- -----------------
    2    m^2 + n^2 + mn
    
For 21.8° => 13/14    9.43° => 73/74    1.05° => 5953/5954    1.16° =>  4873/4874
    (1,2)=(m,n)       (3,4)             (31,32)               (28,29)
    
    3.89° => 433/434  2.65° => 937/938  1.47° => 3037/3038    0.987° => 6733/6734 
    (8,9)             (12,13)           (22,23)               (33,34) or 0.99°
"""
###############################################################################

lattice_a1 = np.array([c*d,0]) #translational vector in x direction for 4-atom basis
lattice_a2 = np.array([0,3*d]) #translational vector in y direction for 4-atom basis


if stack_conf == 'AA':
    layer1_sub_a = np.array([[0,0],[c*d/2,3*d/2]]) #unit cell layer1 sublattice A
    layer1_sub_b = np.array([[0,d],[c*d/2,5*d/2]]) #unit cell layer1 sublattice B   
    layer2_sub_a = np.array([[0,0],[c*d/2,3*d/2]]) #unit cell layer2 sublattice A
    layer2_sub_b = np.array([[0,d],[c*d/2,5*d/2]]) #unit cell layer2 sublattice B

elif stack_conf == 'AB':
    layer1_sub_a = np.array([[0,0],[c*d/2,3*d/2]])    #unit cell layer1 sublattice A
    layer1_sub_b = np.array([[0, d],[c*d/2,5*d/2]])   #unit cell layer1 sublattice B
    layer2_sub_a = np.array([[0,-d],[c*d/2,3*d/2-d]]) #unit cell layer2 sublattice A
    layer2_sub_b = np.array([[0,0],[c*d/2,5*d/2-d]])  #unit cell layer2 sublattice B
    

n = 80  #number of vector for y direction
m = n*2  #number of vector for x direction
# 定義晶格的大小和範圍
x_range = np.arange(-m, m) * lattice_a1[0]
y_range = np.arange(-n, n) * lattice_a2[1]

size = x_range.shape[0]*y_range.shape[0]    #honeycomb sheet size, namely number of unitcell
lattice_layer1_sub_a = np.zeros((size*2,2),dtype=np.float64) #unrotated layer1 sublattice A coordinate
lattice_layer1_sub_b = np.zeros((size*2,2),dtype=np.float64) #unrotated layer1 sublattice B coordinate
lattice_layer2_sub_a = np.zeros((size*2,2),dtype=np.float64) #unrotated layer2 sublattice A coordinate
lattice_layer2_sub_b = np.zeros((size*2,2),dtype=np.float64) #unrotated layer2 sublattice B coordinate

rot_mat1 = np.array([[np.cos(t1/2),-np.sin(t1/2)],  #for layer 1
                    [np.sin(t1/2), np.cos(t1/2)]])

rot_mat2 = np.array([[np.cos(t1/2),np.sin(t1/2)],   #for layer 2
                    [-np.sin(t1/2),np.cos(t1/2)]])

# 使用meshgrid生成網格坐標矩陣
X, Y = np.meshgrid(x_range, y_range)

lattice_layer1_sub_a[:,0] = (layer1_sub_a[:,0,np.newaxis] + X.flatten()).flatten()
lattice_layer1_sub_a[:,1] = (layer1_sub_a[:,1,np.newaxis] + Y.flatten()).flatten()
lattice_layer1_sub_b[:,0] = (layer1_sub_b[:,0,np.newaxis] + X.flatten()).flatten()
lattice_layer1_sub_b[:,1] = (layer1_sub_b[:,1,np.newaxis] + Y.flatten()).flatten()

# Can also write like this way, but slight slow, cost more about 0.0003 second 
#lattice_layer1_sub_a = (layer1_sub_a[:, np.newaxis] + np.array([X.flatten(), Y.flatten()]).T).reshape(-1, 2)

lattice_layer2_sub_a[:,0] = (layer2_sub_a[:,0,np.newaxis] + X.flatten()).flatten()
lattice_layer2_sub_a[:,1] = (layer2_sub_a[:,1,np.newaxis] + Y.flatten()).flatten()
lattice_layer2_sub_b[:,0] = (layer2_sub_b[:,0,np.newaxis] + X.flatten()).flatten()
lattice_layer2_sub_b[:,1] = (layer2_sub_b[:,1,np.newaxis] + Y.flatten()).flatten()


###############################################################################
###
### Let's begin to creat our twisted bilayer graphene
###
###############################################################################
rot_layer1_sub_a = np.dot(lattice_layer1_sub_a,rot_mat1)
rot_layer1_sub_b = np.dot(lattice_layer1_sub_b,rot_mat1)
rot_layer2_sub_a = np.dot(lattice_layer2_sub_a,rot_mat2)
rot_layer2_sub_b = np.dot(lattice_layer2_sub_b,rot_mat2)
 
###############################################################################
###
### Find out the coincident atoms in supercell 
###
###############################################################################

time_2 = time.time()
round_digit = 12 #Significant Figures for np.around()
rot_layer1_sub_a = np.around(rot_layer1_sub_a,round_digit)
rot_layer1_sub_b = np.around(rot_layer1_sub_b,round_digit)
rot_layer2_sub_a = np.around(rot_layer2_sub_a,round_digit)
rot_layer2_sub_b = np.around(rot_layer2_sub_b,round_digit)

rot_layer1 = np.concatenate((rot_layer1_sub_a, rot_layer1_sub_b))
rot_layer2 = np.concatenate((rot_layer2_sub_a, rot_layer2_sub_b))

nrows, ncols = rot_layer1.shape
dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [rot_layer1.dtype]}

#Find the same coordinate in three layers and denote them as black dots
coincident_12 = np.intersect1d(rot_layer1.view(dtype), rot_layer2.view(dtype)) #combine layer1 and layer2 first
#F = np.intersect1d(E.view(dtype), C.view(dtype)) #combine E and C after
# This last bit is optional if you're okay with "F" being a structured array...

#coincident atom coordinates in (x,y) form
coincident_12 = coincident_12.view(rot_layer1.dtype).reshape(-1, ncols)

 
def plot_structure():
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8.5)
    ax.scatter(rot_layer1_sub_a[:,0], rot_layer1_sub_a[:,1], color="red",s=5)   #plot layer1 sublattice A
    ax.scatter(rot_layer1_sub_b[:,0], rot_layer1_sub_b[:,1], color="green",s=5) #plot layer1 sublattice B
    ax.scatter(rot_layer2_sub_a[:,0], rot_layer2_sub_a[:,1], color="blue",s=5)   #plot layer2 sublattice A
    ax.scatter(rot_layer2_sub_b[:,0], rot_layer2_sub_b[:,1], color="orange",s=5) #plot layer2 sublattice B
    ax.scatter(0,0,color="black",s=15)    #denote the origin(0,0)
    ax.set_title("Twisted Trilayer Graphene Structure (angle ~ %s°)"%(angle1))
    ax.set_aspect("equal")
    ax.scatter(coincident_12[:,0],coincident_12[:,1],color="black",s=15) #plot coincident atoms
    plt.show()
    
#plot_structure()
###############################################################################
###
### Let's  begin to find the 4-vertex of our supercell 
###
###############################################################################

#define a funciton for finding element which is closest to a specific point
def find_nearest_vector(array, value):
    idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
    return array[idx]

#contain the commands relate finding vertex of supercell
def find_vertex():
    pass
#Find the (0,0) or the closest point as start point of the supercell
vertex = np.array([])
pt = np.array([0,0])
vertex = np.append(vertex,find_nearest_vector(coincident_12,pt)) #find closest O(0,0) point
print('vertex =',vertex)
#atom O as the supercell's reference point, very important!!!
atomO = vertex

###############################################################################
###
### Find the candidate vertex atoms, such as: A', B, C  
###
###############################################################################
leftt_O_max = coincident_12[np.where(coincident_12[:,0]<vertex[0])][:,0].max() #a maximum number in the  left of the atomO's x coordinate
right_O_min = coincident_12[np.where(coincident_12[:,0]>vertex[0])][:,0].min() #a minimum number in the right of the atomO's x coordinate

test1 = coincident_12[np.where(coincident_12[:,0]<vertex[0])] #for atomC 
test2 = coincident_12[np.where(coincident_12[:,0]>vertex[0])] #for atomB

test_array1 = test1[np.where(test1[:,0]==leftt_O_max)]
test_array2 = test2[np.where(test2[:,0]==right_O_min)]
test_y1 = test_array1[np.where(test_array1[:,1]>vertex[1])][:,1].min()
test_y2 = test_array2[np.where(test_array2[:,1]>vertex[1])][:,1].min()

#atom B  for test whether the supercell is diamond or hexagon
Bx = test_array2[np.where(test_array2[:,1]>vertex[1])][:,0].min()
By = test_array2[np.where(test_array2[:,1]>vertex[1])][:,1].min()
atomB = np.array([Bx,By])
#atom A' for test whether the supercell is diamond or hexagon
test3 = coincident_12[np.where(coincident_12[:,0]==vertex[0])] #for atomAp
Apy = test3[np.where(test3[:,1]>vertex[1])][:,1].min()
Apx = vertex[0]
atomAp = np.array([Apx,Apy])
#atom C  for test whether the supercell is diamond or hexagon
Cx = test_array1[np.where(test_array1[:,1]>vertex[1])][:,0].max()
Cy = test_array1[np.where(test_array1[:,1]>vertex[1])][:,1].min()
atomC = np.array([Cx,Cy])

time_3 = time.time()
print(f'construct lattice cost time: {time_3-time_1} s')
###############################################################################
###
### Create some useful function ~ ~ ~ 
###
###############################################################################

#Plotting the supercell 
def plot_unitcell(vertex,atomb,atoma,atomc):
    vertex = np.append(vertex,atomb)
    vertex = np.append(vertex,atoma)
    vertex = np.append(vertex,atomc)
    vertexk = vertex.reshape(4,2) #reshaped vertex
    unitcell_x = [vertexk[0,0],vertexk[1,0],vertexk[2,0],vertexk[3,0],vertexk[0,0]]
    unitcell_y = [vertexk[0,1],vertexk[1,1],vertexk[2,1],vertexk[3,1],vertexk[0,1]]
    plt.plot(unitcell_x,unitcell_y,c='C0')
    return vertex, vertexk


#Counting atoms in supercell
def count_atom_num(atomo,atomb,atoma,atomc):
    
    polygon = Polygon([atomo, atomb, atoma, atomc])
    
    # create temporary rot_layer1&2 only contain atom's y coordinate > -0.1 for saving search time 
    temp_rot_layer1_sub_a = rot_layer1_sub_a[((rot_layer1_sub_a[:,1]>-0.1) & (rot_layer1_sub_a[:,0]>atomc[0]-0.1) & (rot_layer1_sub_a[:,0]<atomb[0]+0.1))] 
    temp_rot_layer1_sub_b = rot_layer1_sub_b[((rot_layer1_sub_b[:,1]>-0.1) & (rot_layer1_sub_b[:,0]>atomc[0]-0.1) & (rot_layer1_sub_b[:,0]<atomb[0]+0.1))]
    temp_rot_layer2_sub_a = rot_layer2_sub_a[((rot_layer2_sub_a[:,1]>-0.1) & (rot_layer2_sub_a[:,0]>atomc[0]-0.1) & (rot_layer2_sub_a[:,0]<atomb[0]+0.1))]
    temp_rot_layer2_sub_b = rot_layer2_sub_b[((rot_layer2_sub_b[:,1]>-0.1) & (rot_layer2_sub_b[:,0]>atomc[0]-0.1) & (rot_layer2_sub_b[:,0]<atomb[0]+0.1))]

    
    # This way is more concise than above one but less clear,just restore two lines
    # Use list comprehension to filter points that are contained within the polygon
    #in_supercell_layer1_sub_a.extend([point for point in rot_layer1_sub_a if polygon.contains(Point(point[0], point[1]))])
    #in_supercell_layer2_sub_b.extend([point for point in rot_layer2_sub_b if polygon.contains(Point(point[0], point[1]))])

    l1a = np.array([point for point in temp_rot_layer1_sub_a if polygon.contains(Point(point[0], point[1]))])
    l1b = np.array([point for point in temp_rot_layer1_sub_b if polygon.contains(Point(point[0], point[1]))])
    l2a = np.array([point for point in temp_rot_layer2_sub_a if polygon.contains(Point(point[0], point[1]))])
    l2b = np.array([point for point in temp_rot_layer2_sub_b if polygon.contains(Point(point[0], point[1]))])
    
    #for AA stacking: atomo should be included in in_supercell_layer1_sub_A and in_supercell_layer2_sub_A
    if stack_conf == 'AA':
        if n1 == m1:
            l1a = atomo[np.newaxis,:]
            l2a = atomo[np.newaxis,:]
        else:
            l1a = np.vstack((atomo, l1a))
            l2a = np.vstack((atomo, l2a))
    
    #for AB stacking: atomo should be included in in_supercell_layer1_sub_A and in_supercell_layer2_sub_B
    elif stack_conf == 'AB':
        l1a = np.vstack((atomo, l1a))
        l2b = np.vstack((atomo, l2b))
    
    del temp_rot_layer1_sub_a,temp_rot_layer1_sub_b,temp_rot_layer2_sub_a,temp_rot_layer2_sub_b
    
    #print("total atoms =",len(in_supercell_1)+len(in_supercell_2))
    #print("layer1 atoms =",len(in_supercell_1))
    #print("layer2 atoms =",len(in_supercell_2))
    
    #add z component to each atom
    l1a = np.hstack((l1a,np.full((l1a.shape[0], 1), h1)))
    l1b = np.hstack((l1b,np.full((l1b.shape[0], 1), h1)))
    l2a = np.hstack((l2a,np.full((l2a.shape[0], 1), h2)))
    l2b = np.hstack((l2b,np.full((l2b.shape[0], 1), h2)))
    
    total_num =  l1a.shape[0]+l1b.shape[0]+l2a.shape[0]+l2b.shape[0]
    '''
    plt.scatter(l1a[:,0],l1a[:,1],color="C0",s=15) #C0: light blue 
    plt.scatter(l1b[:,0],l1b[:,1],color="C1",s=15) #C1: orange
    plt.scatter(l2a[:,0],l2a[:,1],color="C2",s=15) #C2: green
    plt.scatter(l2b[:,0],l2b[:,1],color="C3",s=15) #C3: red
    '''
    l1 = np.array(atomb) - np.array(atomo)
    l2 = np.array(atomc) - np.array(atomo)
    #print('vector length a1:',np.linalg.norm(L1),"nm")
    #print('vector length a2:',np.linalg.norm(L2),"nm")
    
    return total_num,l1a,l1b,l2a,l2b


#cann= count_atom_num(atomO,atomB,atomAp,atomC)

def distance_matrix():
    
    atoms = np.vstack((can[1:]))
    distances_vec_matrix = atoms[:, np.newaxis] - atoms
    distances_sca_matrix = np.linalg.norm(atoms[:, np.newaxis] - atoms, axis=2)
    #np.savez('arrays.npz', array1=distances_vec_matrix, array2=distances_sca_matrix)
    np.save('distance_matrix.npy', distances_vec_matrix)


def neighbor_list(atomo,atomb,atoma,atomc):
    
    cans = count_atom_num(atomo,atomb,atoma,atomc) # need to know atom coordinate in supercell
    in_supercell = np.vstack((cans[1:]))   # concatenate cans[1] to cans[end] as a large array
    l1 = np.array(atomb) - np.array(atomo) # translational vector of supercell lattice vector L1
    l2 = np.array(atomc) - np.array(atomo)
    neighbor_list = np.zeros((1,2))
    
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            neighbor_list = np.vstack((neighbor_list,i*l1+j*l2+in_supercell))
            
    neighbor_list = neighbor_list[1:]
    return neighbor_list
    
    pass



#Counting atoms in supercell with cutoff radius enlarging range
def tb_model(atomo,atomb,atoma,atomc,kx,ky,sp_zm=0,b_mag=0,beta_d=0,strain_m=1):
    #cans = count_atom_num(atomo,atomb,atoma,atomc) # need to know atom coordinate in supercell
    cans = can
    l1 = np.array(atomb) - np.array(atomo)
    l2 = np.array(atomc) - np.array(atomo)
    L1 = np.array([l1[0], l1[1], 0])  
    L2 = np.array([l2[0], l2[1], 0])
    noa= cans[1].shape[0] # number of atoms in layer1_sublattice A, for constructing Hamiltonian
    nol= len(cans)-1  #number of layer or type of sublattice  of the system
    
    # Basic parameters appear in the Hamiltonian
    a0 = 0.142        #nm, bond length
    d0 = 0.335        #nm, interlayer distance
    Vpppi0 = -2.7     #eV, default = -2.7
    Vppsg0 = 0.48     #eV
    delta  = 0.184*0.246  #nm, decay length of the transfer integral
    cutoff_r = 4*a0+0.01  #cutoff distance for hopping, is d=4*a0 parameter in Koshino's paper
    

    '''
    hopping energy formula:
        
    -t(Ri,Rj) = Vpppi*[1-((Ri-Rj)•ez/cutoff_r)^2] + Vppsg*[((Ri-Rj)•ez/cutoff_r)^2]
    
    Vpppi = Vpppi0*exp(-(cutoff_r-a0)/delta)
    Vppsg = Vppsg0*exp(-(cutoff_r-d0)/delta)
    
    H = 𝚺ij t(Ri,Rj)*|𝛙i><𝛙j| + h.c.
    
    '''
    
    def hopping_t(Ri,Rj,RiRj):
        #RiRj  = np.linalg.norm(Ri-Rj) # is equal to parameter "d" in Koshino's paper
        ez = np.array([0,0,1])
        Vpppi = Vpppi0*np.exp(-(RiRj-a0)/delta)
        Vppsg = Vppsg0*np.exp(-(RiRj-d0)/delta)
        if RiRj > cutoff_r:
            t = 0
        elif abs(RiRj - 0) < 0.0001:
            t = 0     # if onsite = 0 means no onsite
        else:
            if strain_m == 1:
                t = Vpppi*(1-(np.dot((Ri-Rj),ez)/RiRj)**2) + Vppsg*((np.dot((Ri-Rj),ez)/RiRj)**2)
            else:
                #this term is strain term: np.exp(-beta_d*(RiRj_strain/a0-1))
                RiRj_s = np.linalg.norm((Ri-Rj)*np.array([strain_m,strain_m,1]))
                t = Vpppi*(1-(np.dot((Ri-Rj),ez)/RiRj)**2)*np.exp(-beta_d*(RiRj_s/a0-1)) + Vppsg*((np.dot((Ri-Rj),ez)/RiRj)**2)
        return t
    
    def onsite_e_field(noa,ls_index_e,onsite_l1,onsite_l2): #onsite: electric field
        diagnal = np.hstack((np.full(noa*ls_index_e, onsite_l1), np.full(noa*ls_index_e, onsite_l2)))
        diagonal_matrix = np.diag(diagnal)
        return diagonal_matrix
    
    def onsite_stagger_potential(noa,onsite_a,onsite_b): #onsite: staggered potential
        onsite_l1_a  = np.full(noa,onsite_a)
        onsite_l1_b  = np.full(noa,onsite_b)
        diagnal = np.hstack((onsite_l1_a,onsite_l1_b,onsite_l1_a,onsite_l1_b))
        diagonal_matrix = np.diag(diagnal)
        return diagonal_matrix
    
    def moire_potential(cans):
        
        I = complex(0, 1)
        V = 6.6 #default: 6.6
        Psi = -94*np.pi/180 #default: -94*np.pi/180
        
        def Rot(theta):
            return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        b1 = np.array([4*np.pi/np.sqrt(3), 0])
        b2 = Rot(np.pi/3).dot(b1)
        b3 = Rot(np.pi/3).dot(b2)
        b4 = Rot(np.pi/3).dot(b3)
        b5 = Rot(np.pi/3).dot(b4)
        b6 = Rot(np.pi/3).dot(b5)
        
        Vb1 = V*np.exp( I*Psi)
        Vb2 = V*np.exp(-I*Psi)
        Vb3 = Vb1
        Vb5 = Vb1
        Vb4 = Vb2
        Vb6 = Vb2
        
        def MoirePotential(rx, ry):
            r = np.array([rx, ry])
            mp = Vb1*np.exp(I*b1.dot(r))+Vb2*np.exp(I*b2.dot(r))+Vb3*np.exp(I*b3.dot(r))+\
                 Vb4*np.exp(I*b4.dot(r))+Vb5*np.exp(I*b5.dot(r))+Vb6*np.exp(I*b6.dot(r))
            return mp
        
        mp1 = MoirePotential(cans[1][:,0],cans[1][:,1])
        mp2 = MoirePotential(cans[2][:,0],cans[2][:,1])
        mp3 = MoirePotential(cans[3][:,0],cans[3][:,1])
        mp4 = MoirePotential(cans[4][:,0],cans[4][:,1])
        mp = np.array([mp1,mp2,mp3,mp4]).flatten()
        #print('mp=',mp[0][0])
        
        '''
        fig,ax = plt.subplots()
        for i in range(1,5):
            x = can[i][:,0]
            y = can[i][:,1]
            X, Y = np.meshgrid(x, y)
            #print(X,Y)
            #mp1_grid = MoirePotential(X, Y)
            
            mp1_grid = MoirePotential(X.flatten(), Y.flatten())
            mp1_grid = mp1_grid.reshape(X.shape)
            #print(X.shape,mp1_grid.shape)
            
            #plt.contourf(X, Y, mp1_grid, 8, cmap='viridis')
            plt.scatter(x, y, s=mp[i-1])
        #plt.colorbar()
        #plt.axes().set_aspect('equal')
        plt.show()
        '''
        return mp
    
    Ham = np.zeros((kx.shape[0],noa*nol,noa*nol), dtype=complex)
    #h_matrix = np.zeros((noa*nol,noa*nol), dtype=complex)

    def construct_ham(ls_index_s,ls_index_h): # hopping for atoms in layer_i sublattice_alpha
        #ls_index_s stands for index for select_atom cans[i], where i = 1,2,3,4...
        #ls_index_h stands for index for hopping cans[i], where i = 1,2,3,4...
        '''
        new_cans = np.zeros((9,noa,3))
        for x in range(-1,2,1):     # x times L1
            for y in range(-1,2,1): # y times L2
                #for i in range(cans[ls_index_h].shape[0]):   # shift layer1 sublattice A's atom
                new_cans[(x+1)*3+(y+1)] = cans[ls_index_h] + x*L1 + y*L2
        #print('new_i=',new_i)
        '''
        
        x_values = np.arange(-1, 2)
        y_values = np.arange(-1, 2)
        
        # 生成所有 x 和 y 值的組合
        X, Y = np.meshgrid(x_values, y_values)
        shifts = np.column_stack((X.ravel(), Y.ravel()))
        
        # 利用廣播機制進行向量化計算
        new_cans = cans[ls_index_h] + shifts.dot(np.vstack((L1, L2)))[:, np.newaxis]
        
        for k in range(noa):    # k is the selected atom
            select_atom = cans[ls_index_s][k]   
            
            for ii in range(9): #The 3x3 grid consists of 9 cells, and from (-1,-1), (-1,0) ... to (1,1), objects that can hop are selected.
                for i in range(noa):

                    distance = np.linalg.norm(select_atom - new_cans[ii][i]) #is equal to vector "d" in Koshino's paper

                    if distance < cutoff_r:
                        Rij = select_atom - new_cans[ii][i] 
                        hopping_energy = hopping_t(select_atom,new_cans[ii][i],distance)
                        for kxy in range(kx.shape[0]):
                            Ham[kxy][k+noa*(ls_index_s-1)][i+noa*(ls_index_h-1)] += hopping_energy*np.exp(1j*(Rij[0]*kx[kxy]+Rij[1]*ky[kxy]))
                
    construct_ham(1,1)  
    construct_ham(1,2)  
    construct_ham(1,3)
    construct_ham(1,4)
    
    construct_ham(2,1)  
    construct_ham(2,2)  
    construct_ham(2,3)
    construct_ham(2,4)
    
    construct_ham(3,1)  
    construct_ham(3,2)  
    construct_ham(3,3)
    construct_ham(3,4)
    
    construct_ham(4,1)  
    construct_ham(4,2)  
    construct_ham(4,3)
    construct_ham(4,4)

    
    onsite_e_field = onsite_e_field(noa,2,0.02,0)
    onsite_s_potnt = onsite_stagger_potential(noa,1,-1)
    #Ham += onsite_e_field  # add electric field 
    #Ham += onsite_s_potnt  # add staggered potential
    
    #MP = moire_potential(cans)
    #MP[(MP.shape[0]//2):] = 0  # top layer with MP, bottom layer not
    #Ham += MP
   

    def spin_pn(h,b): # h is Hamiltonian

        pauli_z = np.array([[1, 0], [0,-1]]) # Pauli matrix, sigma_z, diagonal_multiplier
        pauli_i = np.array([[1, 0], [0, 1]]) # Identity matrix, off_diagonal_multiplier

        diagonal = np.diag(h)  # get diagonal term, an 1d array
        diagonal = diagonal*np.identity(noa*nol) # generate diagonal matrix using diagonal term
        off_diagonal = h - diagonal*np.identity(noa*nol) # generate off-diagonal matrix

        new_dia = np.kron(h, b*pauli_z)     # all the diagonal term times pauli_sigma_z matrix
        new_off_dia = np.kron(h, pauli_i)   # all the off-diagonal term times identity matrix
        new_matrix = new_dia + new_off_dia  # combine two matrix as unity
        return new_matrix

    if sp_zm != 0:  #open spin dependent onsite term
        Ham = spin_pn(Ham,sp_zm)
    
    def zeeman(h,b_m):
        
        pauli_i = np.array([[1, 0], [0, 1]]) # Identity matrix, off_diagonal_multiplier
        new_ham = np.kron(pauli_i, h[:])
    
        N = new_ham[1].shape[0] // 2
        added_zeeman_matrix = np.zeros((2*N,2*N))
        added_zeeman_matrix[:N, :N] += b_m * np.eye(N)
        added_zeeman_matrix[N:, N:] -= b_m * np.eye(N)
        new_ham = new_ham + added_zeeman_matrix

        return new_ham
    
    if b_mag != 0:  #open magnetic field for Zeeman term
        Ham = zeeman(Ham,b_mag)
    

    #Ham_t = Ham.conj().T
    #Ham += Ham_t
    Ham_t = Ham.conj().transpose(0, 2, 1)
    Ham += Ham_t
    #print(Ham)  # check Hamiltonian
    
    return Ham 


def solve_eig(Ham,shift=0):
    '''   
    eigenvalue,featurevector=np.linalg.eigh(Ham)
    eig_vals_sorted = np.sort(eigenvalue)
    #e=cp.asnumpy(eig_vals_sorted)
    e=eig_vals_sorted
    '''
    #print(Ham.dtype)
    
    eigenvalue,featurevector = np.linalg.eigh(Ham)
    e = eigenvalue
    #e,i = torch.sort(eigenvalue)
    return e/2 - 0.783427 +shift -0.00378 # shift to Fermi level: 0 eV


def k_path_bnd(lattice_const, vec_angle, k_sec_num):
    
    kD = 4*np.pi/(3*lattice_const)
    KDens = kdens       # density of k points in path for band structure, default = 10
    
    if k_sec_num == 4:  # Kp G M K G, 4 k-path sections
        KptoG = np.arange(1, 0, -1/KDens)
        GtoM  = np.arange(0, np.sqrt(3)/2, 1/KDens)
        MtoK  = np.arange(0, -1/2, -1/KDens)
        KtoG  = np.arange(1, 0-1/KDens/10, -1/KDens)
        
        AllK  =  len(KptoG) + len(GtoM) + len(MtoK) + len(KtoG)
        k_path1 = np.zeros((AllK,2)) # for vec_angle > 90
        k_path2 = np.zeros((AllK,2)) # for vec_angle < 90
        Kpaths_coord = [0, len(KptoG), len(KptoG)+len(GtoM),len(KptoG)+len(GtoM)+len(MtoK), AllK-1]
        
        for i, k in enumerate(KptoG):
            k_path1[i] = [np.sqrt(3)/2 * k * kD, 1/2 * k * kD]
            k_path2[i] = [1/2 * k * kD, np.sqrt(3)/2 * k * kD]
    
        for i, k in enumerate(GtoM, start=len(KptoG)):
            k_path1[i] = [1 * k * kD,   0 * k * kD]
            k_path2[i] = [np.sqrt(3)/2 * k * kD, 1/2 * k * kD]
        
        for i, k in enumerate(MtoK, start=len(KptoG) + len(GtoM)):
            k_path1[i] = [np.sqrt(3)/2 * kD, k * kD]
            k_path2[i] = [(3/4+(-1/2)*k)*kD, (np.sqrt(3)/4+np.sqrt(3)/2*k)*kD]
        
        for i, k in enumerate(KtoG, start=len(KptoG) + len(GtoM) + len(MtoK)):
            k_path1[i] = [-np.sqrt(3)/2 * k * kD, -1/2 * k * kD]
            k_path2[i] = [ 1 * k * kD,    0 * k * kD]
    
    elif k_sec_num == 3:  # G M K G, 3 k-path sections
        GtoM  = np.arange(0, np.sqrt(3)/2, 1/KDens)
        MtoK  = np.arange(0, -1/2, -1/KDens)
        KtoG  = np.arange(1, 0-1/KDens/10, -1/KDens)
        
        AllK  =  len(GtoM) + len(MtoK) + len(KtoG)
        k_path1 = np.zeros((AllK,2)) # for vec_angle > 90
        k_path2 = np.zeros((AllK,2)) # for vec_angle < 90
        Kpaths_coord = [0, len(GtoM),len(GtoM)+len(MtoK), AllK-1]
    
        for i, k in enumerate(GtoM):
            k_path1[i] = [1 * k * kD,   0 * k * kD]
            k_path2[i] = [np.sqrt(3)/2 * k * kD, 1/2 * k * kD]
        
        for i, k in enumerate(MtoK, start=len(GtoM)):
            k_path1[i] = [np.sqrt(3)/2 * kD, k * kD]
            k_path2[i] = [(3/4+(-1/2)*k)*kD, (np.sqrt(3)/4+np.sqrt(3)/2*k)*kD]
        
        for i, k in enumerate(KtoG, start=len(GtoM) + len(MtoK)):
            k_path1[i] = [-np.sqrt(3)/2 * k * kD, -1/2 * k * kD]
            k_path2[i] = [ 1 * k * kD,    0 * k * kD]
    
    if vec_angle > 90:
        k_path = k_path1
    else:
        k_path = k_path2    
    
    return AllK, k_path, Kpaths_coord


def plot_band_structure(cell_atom, AllK, E, Kpaths, y_range_list=None, save_pic=None):
    
    for j in range(0,cell_atom):   
        #plt.plot(np.arange(AllK), (E[:,j]), c='k', linewidth=2)
        if j == can[0]//2 - 1:
            plt.plot(np.arange(AllK), (E[:,j]), c='b', linewidth=2)
        elif j == can[0]//2:
            plt.plot(np.arange(AllK), (E[:,j]), c='r', linewidth=2)
        else:
            plt.plot(np.arange(AllK), (E[:,j]), c='k', linewidth=2)
    
    for i in Kpaths:    
        plt.axvline(x=i,ls=':')
        
    plt.axhline(y=0,ls=':',c='g')
    if len(Kpaths) == 5:
        plt.xticks(Kpaths, ("K$^‘$", '$\Gamma$', 'M', 'K','$\Gamma$'), fontsize=18)
    elif len(Kpaths) == 4:
        plt.xticks(Kpaths, ('$\Gamma$', 'M', 'K','$\Gamma$'), fontsize=18)
        
    plt.yticks(fontsize=10)
    plt.ylabel('E(eV)', fontsize=12)
    
    # 判斷 y_range_list 是否為 None
    if y_range_list is not None:
        plt.ylim(y_range_list)
        
    # 設定縱座標刻度標籤
    #plt.yticks(np.arange(y_range_list[0], y_range_list[1], 0.1))
    plt.title(f'{angle1}° Twisted bilayer graphene')
    if save_pic is not None:
        plt.savefig(f'TBG_{int(angle1*100)}_bnd.png',dpi=200)
        print('Image saved successfully')
        
    eu, el = 1.2, -1.2 
    #plt.ylim(el, eu)
    plt.show()
    

def plot_tb_band(atomo,atomb,atoma,atomc,kdens=kdens,plotb=plotb):
     
    # if open Zeeman term, cell_atom = count_atom_num()*2
    mag = 0     # for Zeeman term 
    sp_mag = 0  # for spin up/down potential
    if mag == 0:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0] # number of atoms in supercell
        #cell_atom = 1300
        print(f"number of atoms in supercell = {cell_atom}")
    else:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0]*2 # number of atoms in supercell
        print(f"number of atoms in supercell = {int(cell_atom/2)}")
    lattice_const = np.linalg.norm(np.array(atomb)-np.array(atomo))
    
    # determine cell is 120 or 60 degree diamond
    # path: K' G M K G 
    cell_a1 = np.array(atomb)-np.array(atomo)
    cell_a2 = np.array(atomc)-np.array(atomo)
    vec_angle = np.arccos(np.dot(cell_a1,cell_a2)/(np.linalg.norm(cell_a1)*np.linalg.norm(cell_a2)))/np.pi*180
    
    path_num = 4 # control k-path sections, 4 means path: K' G M K G; 3 means path: G M K G
    AllK,k_path, Kpaths = k_path_bnd(lattice_const, vec_angle, path_num)
    E  = np.zeros((AllK,cell_atom), dtype = np.complex64) #np.complex128

    s_time = time.time()  
    hamiltonian = tb_model(atomo,atomb,atoma,atomc,k_path[:,0],k_path[:,1],b_mag=mag)
    e_time = time.time()
    print(f'Total construct Hamiltonian cost {e_time-s_time} s')
    
    
    for i in range(AllK):
        s_time = time.time() 
        E[i] = solve_eig(hamiltonian[i],shift=-0.015+0.0148+0.00019-0.0102) 
        e_time = time.time() 
        if i%20==0:
            print(f"{i:5d}_th k cost time = {(e_time-s_time):.6f}")
    print(type(E))
    
    # store Eigenvalue and K path
    # Eigv_material name(include twisted angle)_stacking configuration_k path number_k point number in each path
    #Kpaths = 
    Klabel = ["K$^‘$", '$\Gamma$', 'M', 'K','$\Gamma$']
    tangle = angle1
    name   = f'TBG_{int(angle1*100)}_{stack_conf}'
    
    path = '/home/wxkao/Documents/Python_tool/Eigv_AllK_npy/'
    #np.save(path + f'Eigv_TBG_{int(angle1*100)}_{stack_conf}_{5}_{kdens}.npy', E)
    #np.savez(path + f'AllK_TBG_{int(angle1*100)}_{stack_conf}_{5}_{kdens}.npz', Kpaths, Klabel, tangle, name)
    
    
    # if plotb is open (=1), then plot band structure; else, don't plot
    if plotb == 1:
        plot_band_structure(cell_atom, AllK, E, Kpaths, y_range_list=None, save_pic=None)
            
    
def plot_tb_band2(atomo,atomb,atoma,atomc,kdens=kdens,plotb=plotb):
    
    # if open Zeeman term, cell_atom = count_atom_num()*2
    mag = 0     # for Zeeman term 
    sp_mag = 0  # for spin up/down potential
    if mag == 0:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0] # number of atoms in supercell
        #cell_atom = 1300
        print(f"number of atoms in supercell = {cell_atom}")
    else:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0]*2 # number of atoms in supercell
        print(f"number of atoms in supercell = {int(cell_atom/2)}")
    lattice_const = np.linalg.norm(np.array(atomb)-np.array(atomo))    
    
    # determine cell is 120 or 60 degree diamond
    cell_a1 = np.array(atomb)-np.array(atomo)
    cell_a2 = np.array(atomc)-np.array(atomo)
    vec_angle = np.arccos(np.dot(cell_a1,cell_a2)/(np.linalg.norm(cell_a1)*np.linalg.norm(cell_a2)))/np.pi*180

    AllK,k_path, Kpaths = k_path_bnd(lattice_const, vec_angle,4)
    E  = np.zeros((AllK,cell_atom), dtype = np.complex64) #np.complex128
            

    s_time = time.time()  
    strain_m = 1.10 
    hamiltonian = tb_model(atomo,atomb,atoma,atomc,k_path[:,0],k_path[:,1],b_mag=mag,beta_d=3.37,strain_m=strain_m)
    e_time = time.time()
    print(f'Total construct Hamiltonian cost {e_time-s_time} s')
    
    
    for i in range(AllK):
        s_time = time.time() 
        E[i] = solve_eig(hamiltonian[i],shift=0.73+0.03-0.0277+7.9*10**(-5)) #0.714328
        e_time = time.time() 
        if i%10==0:
            print(f"{i:5d}_th k cost time = {(e_time-s_time):.6f}")
    print(type(E))
    
    # store Eigenvalue and K path
    # Eigv_material name(include twisted angle)_stacking configuration_k path number_k point number in each path
    Klabel = ["K$^‘$", '$\Gamma$', 'M', 'K','$\Gamma$']
    tangle = angle1
    name   = f'TBG_{int(angle1*100)}_{stack_conf}'
    
    path = '/home/wxkao/Documents/Python_tool/Eigv_AllK_npy/'
    #np.save(path + f'Eigv_TBG_{int(angle1*100)}_{stack_conf}_{5}_{kdens}_st{strain_m}.npy', E)
    #np.savez(path + f'AllK_TBG_{int(angle1*100)}_{stack_conf}_{5}_{kdens}_st{strain_m}.npz', Kpaths, Klabel, tangle,name)
    
    # plot band structure
    #fig,ax = plt.subplots()

    if plotb == 1:
        for j in range(0,cell_atom):
            plt.plot(np.arange(AllK), (E[:,j]), c='r', linewidth=2)
    
        for i in Kpaths:    
            plt.axvline(x=i,ls=':')
            
        #plt.axhline(y=0,ls=':',c='g')
        #plt.xticks(Kpaths, ("K$^‘$", '$\Gamma$', 'M', 'K','$\Gamma$'), fontsize=18)
        #plt.yticks(fontsize=10)
        #plt.ylabel('E(eV)', fontsize=12)
        #plt.ylim(-0.2,0.2)
        #plt.ylim(-1,1)
        #plt.yticks(np.arange(-0.5, 0.6,0.1))
        #plt.title(f'{angle1}° Twisted bilayer graphene')
        #plt.savefig(f'TBG_{int(angle1*100)}_bnd.png',dpi=200)
        plt.show()
        
    
def plot_tb_band3(atomo,atomb,atoma,atomc,kdens=kdens,plotb=plotb):
     
    # if open Zeeman term, cell_atom = count_atom_num()*2
    mag = 0     # for Zeeman term 
    sp_mag = 0  # for spin up/down potential
    if mag == 0:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0] # number of atoms in supercell
        #cell_atom = 1300
        print(f"number of atoms in supercell = {cell_atom}")
    else:
        cell_atom = count_atom_num(atomo,atomb,atoma,atomc)[0]*2 # number of atoms in supercell
        print(f"number of atoms in supercell = {int(cell_atom/2)}")
    lattice_const = np.linalg.norm(np.array(atomb)-np.array(atomo))
    
    
    # determine cell is 120 or 60 degree diamond
    # path: from Kp G M K G, two or three of them
    cell_a1 = np.array(atomb)-np.array(atomo)
    cell_a2 = np.array(atomc)-np.array(atomo)
    vec_angle = np.arccos(np.dot(cell_a1,cell_a2)/(np.linalg.norm(cell_a1)*np.linalg.norm(cell_a2)))/np.pi*180
    
    #=======================================================================================
    #
    #   It needs to be modified into a version that can draw near any high symmetry point. 
    #
    #=======================================================================================
    '''
    AllK,k_path, Kpaths = k_path_bnd(lattice_const, vec_angle,3)
    E  = np.zeros((AllK,cell_atom), dtype = np.complex64) #np.complex128

    s_time = time.time()  
    hamiltonian = tb_model(atomo,atomb,atoma,atomc,k_path[:,0],k_path[:,1],b_mag=mag)
    e_time = time.time()
    print(f'Total construct Hamiltonian cost {e_time-s_time} s')
    
    
    for i in range(AllK):
        s_time = time.time() 
        E[i] = solve_eig(hamiltonian[i],shift=-0.015+0.0148+0.00019-0.0102) 
        e_time = time.time() 
        print(f"{i:5d}_th k cost time = {(e_time-s_time):.6f}")
    print(type(E))
    
    # store Eigenvalue and K path
    # Eigv_material name(include twisted angle)_stacking configuration_k path number_k point number in each path
    #Kpaths = [0,  len(GtoM),len(GtoM)+len(MtoK), AllK-1]
    Klabel = ['$\Gamma$', 'M', 'K','$\Gamma$']
    tangle = angle1
    name   = f'TBG_{int(angle1*100)}_{stack_conf}'
    
    path = '/home/wxkao/Documents/Python_tool/Eigv_AllK_npy/'
    #np.save(path + f'Eigv_TBG_{int(angle1*100)}_{stack_conf}_{4}_{kdens}.npy', E)
    #np.savez(path + f'AllK_TBG_{int(angle1*100)}_{stack_conf}_{4}_{kdens}.npz', Kpaths, Klabel, tangle, name)

    
    # if plotb is open (=1), then plot band structure; else, don't plot
    if plotb == 1:
        plot_band_structure(cell_atom, AllK, E, Kpaths, y_range_list=None, save_pic=None)
    '''

def density_of_state(atomo,atomb,atoma,atomc,method='GF'):
    
    # create mesh
    # kpoints = 20, calculate for 800 atoms in 47 seconds
    # kpoints = 30, calculate for 800 atoms in 88 seconds
    kpoints = 25 
    mag = 0
    a1 = atomb-atomo
    a2 = atomc-atomO
    
    a_vector = np.array([a1[0:2],a2[0:2]]).T
    print(a_vector)
    b_vector = 2*np.pi*np.linalg.inv(a_vector)
    b1 = b_vector[0][0]
    b2 = b_vector[0][1]
    print(b_vector)
    x, y = np.meshgrid(np.linspace(-b1, b1, kpoints), np.linspace(-b2, b2, kpoints))
    # find the points inside the reciprocal lattice (diamond shape)
    in_diamond = (np.abs(x / b1) + np.abs(y / b2)) <= 1
    
    # find the coordinates of the points in the diamond shape
    kx = x[in_diamond]
    ky = y[in_diamond]
    print('x y shape:', x.shape, y.shape)
    print('kx ky shape:', kx.shape, ky.shape)
    print(' x=', x)
    print('kx=',kx)
    
    s_time = time.time()
    def f(kxx, kyy):
        hamiltonian = tb_model(atomo,atomb,atoma,atomc,kxx,kyy,b_mag=mag,beta_d=3.37,strain_m=1)
        eigenvalue,featurevector = np.linalg.eigh(hamiltonian)
        #e = eigenvalue
        
        #return np.sqrt(1 + 4 * np.cos((np.sqrt(3) * a / 2) * kx) * np.cos(a / 2 * ky) + 4 * np.cos(a / 2 * ky)**2)
        return eigenvalue/2 - 0.793637
        
    def e(kxx, kyy):
        return f(kxx, kyy)
    
    # calculate e1, namely H(kx,ky) and solve eigenvalues
    e1_values = e(kx, ky)
    e_time = time.time()
    print(f'calculate dos cost {e_time-s_time} s')
    
    # Method 1. Histogram to plot DOS
    if method == 'hist': 

        flatten_e = e1_values.flatten()
        print(kx.shape, ky.shape, flatten_e.shape)
        plt.figure()
        #h = plt.hist(flatten_e, bins=300, density=True, orientation='horizontal')
        
        #import seaborn as sns
        # plotting density plot for carat using distplot()
        #sns.distplot(a=flatten_e, bins=300, color='blue',kde = True,hist_kws={"edgecolor": 'blue'})
        
        bins = 150  # number of bins
        h = plt.hist(flatten_e, bins=bins, density=True, orientation='horizontal')
        #plt.clf()   # hide historgram or use: count, bins  = np.histogram(s, 100, normed=True)
        bin_shift = (h[1].max()-h[1].min())/2/bins  # shift to the center of each bin
        #plt.plot(h[1][:-1]+bin_shift,h[0],c='k')
        plt.plot(h[0], h[1][:-1]+bin_shift, c='k')


    # Method 2. Green's Function
    elif method == 'GF': 

        # parameter for Green's Function
        eu = 1.2
        el = -1.2
        Ne = 50
        delta = 0.05
        Fermi_energy_array = np.linspace(el, eu, Ne)  # 计算中取的费米能Fermi_energy组成的数组
        
        # Start calculating Green's Function
        s_time = time.time()
        dos_fn = np.zeros(Fermi_energy_array.shape[0]) 
        for i in range(Fermi_energy_array.shape[0]):
            
            delta_value = 0 #for sum all over k
            for bnd_index in range(e1_values.shape[1]):
                for k in range(e1_values.shape[0]):
                    
                    # To set suitable delta for Dirac delta function, 
                    # we need to know the order of Ef-H(k), so we print them:
                    #print((Fermi_energy_array[i]-e1_values[k][bnd_index]))
                    delta_value += 1/(np.pi) * (delta/((Fermi_energy_array[i]-e1_values[k][bnd_index])**2+delta**2))
            
            dos_fn[i] = delta_value.real
            
        e_time = time.time()
        print(f'calculate DOS using GF cost {e_time-s_time} s')
        
        plt.figure()
        plt.plot(dos_fn, Fermi_energy_array)
        plt.ylim(el, eu)
    
    plt.title('Density of state')
    plt.ylabel('E(eV)')
    plt.xlabel('DOS')
    plt.grid(alpha=0.5)
    plt.show()

    
def Fermi_surface0(atomo,atomb,atoma,atomc):
    
    # create mesh
    # kpoints = 20, calculate for 800 atoms in 47 seconds
    # kpoints = 30, calculate for 800 atoms in 88 seconds
    kpoints = 15 
    mag = 0
    a1 = atomb-atomo
    a2 = atomc-atomO
    
    a_vector = np.array([a1[0:2],a2[0:2]]).T
    print(a_vector)
    b_vector = 2*np.pi*np.linalg.inv(a_vector)
    b1 = b_vector[0][0]
    b2 = b_vector[0][1]
    print(b_vector)
    x, y = np.meshgrid(np.linspace(-b1, b1, kpoints), np.linspace(-b2, b2, kpoints))
    # find the points inside the reciprocal lattice (diamond shape)
    in_diamond = (np.abs(x / b1) + np.abs(y / b2)) <= 1
    
    # construct the points to determine whether they belong to the hexagonal BZ
    k_points = np.vstack((x.flatten(), y.flatten())).T
    
    # find the coordinates of the points in the diamond shape
    kx = x[in_diamond]
    ky = y[in_diamond]
    
    s_time = time.time()
    def f(kxx, kyy):
        hamiltonian = tb_model(atomo,atomb,atoma,atomc,kxx,kyy,b_mag=mag,beta_d=3.37,strain_m=1)
        eigenvalue,featurevector = np.linalg.eigh(hamiltonian)
        #e = eigenvalue
        
        #return np.sqrt(1 + 4 * np.cos((np.sqrt(3) * a / 2) * kx) * np.cos(a / 2 * ky) + 4 * np.cos(a / 2 * ky)**2)
        return eigenvalue/2 - 0.793637
        
    def e(kxx, kyy):
        return f(kxx, kyy)
    
    # calculate e1, namely H(kx,ky) and solve eigenvalues
    #e1_values = e(kx, ky)
    e_time = time.time()
    print(f'calculate dos cost {e_time-s_time} s')
    
    
    xx = np.array([0,b_vector[0][0],(b_vector[0]+b_vector[1])[0],b_vector[1][0],0])
    yy = np.array([0,b_vector[0][1],(b_vector[0]+b_vector[1])[1],b_vector[1][1],0])-b_vector[0][1]
    #plt.plot(xx,yy,c='r')
    #plt.scatter(kx,ky)
    
    #===============================================
    #plot hexagonal BZ for 'horizontal diamond' 
    #===============================================
    kpy = b_vector[0][1]/3
    kpx = b_vector[0][1]/np.sqrt(3)
    
    # first vertex for hexagonal BZ
    vertex = np.array([kpx, kpy])
    
    # 旋轉矩陣 (60度)
    theta = 2 * np.pi / 6
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 計算其餘五個頂點
    vertices = [vertex]
    current_vertex = vertex
    for _ in range(6):
        current_vertex = rotation_matrix @ current_vertex
        vertices.append(current_vertex)
    
    
    polygon = Polygon([point.tolist() for point in vertices[:-1]])
    # 擴展多邊形邊界以包含邊界上的點
    buffered_polygon = polygon.buffer(1e-9)  # 1e-9是擴展的範圍，可以根據需要調整
    inpolygon = np.array([point for point in k_points if buffered_polygon.contains(Point(point[0], point[1]))])
    
    # calculating 2d band 
    e1_values = e(inpolygon[:,0], inpolygon[:,1])
    print('type:',type(e1_values),e1_values.shape,inpolygon.shape)
    
    # select energy range 
    egf = e1_values[(e1_values > 3) & (e1_values < 3.01)]
    indices = np.where((e1_values > 3) & (e1_values < 3.01))
    print('indices=',indices[0])
    #egk = inpolygon[indices][0]
    data = np.zeros(inpolygon.shape[0])
    data[indices[0]] = e1_values[indices]
    print('e1_values[indices]=',e1_values[indices],data)

    
    plt.scatter(inpolygon[:,0], inpolygon[:,1], c=data, cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label='Energy Value')
    # 將頂點轉為numpy陣列
    vertices = np.array(vertices)
    plt.plot(vertices[:,0],vertices[:,1],c='k')
    plt.arrow(0,0,b_vector[0][0],b_vector[0][1],width=0.005,head_width=0.5,color='r',length_includes_head=True,zorder=15)
    plt.arrow(0,0,b_vector[1][0],b_vector[1][1],width=0.005,head_width=0.5,color='r',length_includes_head=True,zorder=15)
    #plt.scatter(k_points[:,0], k_points[:,1])
    #plt.scatter(inpolygon[:,0], inpolygon[:,1])
    plt.axis('equal')
    plt.show()
    


def Fermi_surface1(atomo,atomb,atoma,atomc):
    
    # Define constants
    e0 = 0 
    mag= 0
    kr_grid = 75
    
    
    # Create a grid of kx and ky values
    kxr = np.linspace(-3*np.pi, 3*np.pi, kr_grid)    
    kyr = np.linspace(-3*np.pi, 3*np.pi, kr_grid)
    
    # find the coordinates of the points in the diamond shape
    KX, KY = np.meshgrid(kxr, kyr)
    in_diamond = (np.abs(KX / (3*np.pi)) + np.abs(KY / (3*np.pi))) <= 2
    
    # find the coordinates of the points in the diamond shape
    kx = KX[in_diamond]
    ky = KY[in_diamond]
    
    
    # Define the Hamiltonian function
    def f(kxx, kyy):
        hamiltonian = e0 + tb_model(atomo,atomb,atoma,atomc,kxx,kyy,b_mag=mag,beta_d=3.37,strain_m=1)
        eigenvalue,featurevector = np.linalg.eigh(hamiltonian)
        return eigenvalue/2 - 0.793637
    
    def e(kxx, kyy):
        return f(kxx, kyy)
    
    dummy = 0
    eig_kxy = e(kx,ky)
    for index in range(dummy,28):

        # calculate e1, namely H(kx,ky) and solve eigenvalues
        e1_values = eig_kxy[:,index].reshape(kr_grid, kr_grid)
        
        # Plot the heatmap
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(KX, KY, e1_values, shading='auto', cmap='GnBu')
        #plt.colorbar(label='Energy')
    
        # Add contour lines
        contours = plt.contour(KX, KY, e1_values, colors='k', linewidths=0.5, linestyles= 'dashdot' )
        # text for contour lines
        #plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    
        # Set labels and title
        #plt.xlabel('kx')
        #plt.ylabel('ky')
        #plt.title(f'Fermi surface, {bnd_index}')
        plt.axis('off')
        #plt.axis('eqaul')
        plt.savefig(f'2179_{index}.jpg',dpi=200)
        plt.show()
        plt.close()

def Fermi_surface(atomo,atomb,atoma,atomc):
    
    # Define constants
    e0 = 2.0 
    mag= 0
    kr_grid = 150
    
    
    # Create a grid of kx and ky values
    kxr = np.linspace(-2*np.pi, 2*np.pi, kr_grid)    
    kyr = np.linspace(-2*np.pi, 2*np.pi, kr_grid)
    
    # find the coordinates of the points in the diamond shape
    KX, KY = np.meshgrid(kxr, kyr)
    in_diamond = (np.abs(KX / (3*np.pi)) + np.abs(KY / (3*np.pi))) <= 2
    
    # find the coordinates of the points in the diamond shape
    kx = KX[in_diamond]
    ky = KY[in_diamond]
    
    
    # Define the Hamiltonian function
    def f(kxx, kyy):
        hamiltonian = e0 + tb_model(atomo,atomb,atoma,atomc,kxx,kyy,b_mag=mag,beta_d=3.37,strain_m=1)
        eigenvalue,featurevector = np.linalg.eigh(hamiltonian)
        return eigenvalue/2 - 0.793637
    
    def e(kxx, kyy):
        return f(kxx, kyy)
    
    plt.figure(figsize=(6, 6))
    eig_kxy = e(kx,ky) # store the all kx,ky dependent bands calculation result
    
    for index in range(27):
        bnd_index = index
        # calculate e1, namely H(kx,ky) and solve eigenvalues
        e1_values = eig_kxy[:,bnd_index].reshape(kr_grid, kr_grid)

        # 創建一個布林掩碼，用於選擇數值介於 -0.1 和 0.1 之間的元素
        mask = (e1_values >= e0-0.02) & (e1_values <= e0)
        
        # 將不在範圍內的元素設為 0
        e1_values_filtered = np.where(mask, e1_values, 0)
        # Plot the heatmap
        
        #plt.pcolormesh(KX, KY, e1_values, shading='auto', cmap='GnBu')
        #plt.colorbar(label='Energy')
    
        # Add contour lines
        contours = plt.contour(KX, KY, e1_values_filtered, colors='k', linewidths=0.5, linestyles= 'dashdot' )
        # text for contour lines
        #plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    
    # Set labels and title
    plt.xlabel('kx')
    plt.ylabel('ky')
        #plt.title(f'Fermi surface, {bnd_index}')
        #plt.savefig(f'2179_{index}.jpg',dpi=200)
    plt.axis('equal')
    plt.show()
        #plt.close()
        

#plot supercell lattice vector (with arrow)
def plot_lattice_vec_arrow(atomo,atomb,atomc):
    plt.arrow(atomo[0],atomo[1],atomb[0],atomb[1],width=0.005,head_width=0.05,color='black',length_includes_head=True,zorder=15)
    plt.arrow(atomo[0],atomo[1],atomc[0],atomc[1],width=0.005,head_width=0.05,color='black',length_includes_head=True,zorder=15)

#define POSCAR generator which can generate the POSCAR
#TBG lattice vector
#POSCAR 單位是A，所以我的單位是nm轉A要乘以10
def POSCAR_generator(atomo,atoma,atomb,atomc):    
    
    #Define lattice vertors of supercell
    l1 = np.array(atomb) - np.array(atomo)
    l2 = np.array(atomc) - np.array(atomo)
    L1 = [10*l1[0], 10*l1[1], 0]  
    L2 = [10*l2[0], 10*l2[1], 0]
    L3 = [0, 0, 20]  #z方向 20為考慮真空層的厚度(12A)
    nol= len(can)-1  #number of layer and sublattice
    
    path = f'POSCAR_TBG_{int(angle1*100)}_{stack_conf}'
    with open(path, 'w') as f:
        f.write(f'TBG {stack_conf} stacking with {angle1} twisted angle\n')
        f.write('1.0\n') #scaling factor
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L1[0], L1[1], L1[2]))
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L2[0], L2[1], L2[2]))
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L3[0], L3[1], L3[2]))
        f.write('C\n')
        f.write("%s\n" %(can[0]))
        f.write('C\n')
        s1=time.time()
        #Coordinates for the each atom
        for i in range(1,nol+1): 
            for j in range(len(can[i])):
                f.write(f"{can[i][j][0]*10:12.8f} {can[i][j][1]*10:12.8f} {can[i][j][2]*10:12.8f}\n")
        e1=time.time()
    #print(f'cost time: {e1-s1}')
    
###############################################################################
###
### Categorize the type of supercell and plot the unit cell
###
###############################################################################

def operate_main_fn(atomo,atomb,atoma,atomc, control_switch_list):
    '''
    control_switch_list:
    It is a list that controls whether 8 functions should be executed respectively.
    
    For example:
    control_switch_list = [0,0,0,1,1,1,1,1]   
    This means that the first three functions are not executed, and the last five functions are executed.
    '''
    # Structure related function
    if control_switch_list[0]:
        plot_unitcell(atomo,atomb,atoma,atomc)
    if control_switch_list[1]:
        plot_lattice_vec_arrow(atomo,atomb,atomc)
    if control_switch_list[2]:
        POSCAR_generator(atomo,atoma,atomb,atomc)
        
    if plotb == 1:
        fig,ax = plt.subplots()
        Fermi_surface1(atomo,atomb,atoma,atomc)
        
    # Band structure or DOS related function
    if control_switch_list[4]:
        plot_tb_band(atomo,atomb,atoma,atomc,plotb=plotb)
    if control_switch_list[5]:
        plot_tb_band2(atomo,atomb,atoma,atomc,plotb=plotb)
    if control_switch_list[6]:
        plot_tb_band3(atomo,atomb,atoma,atomc,plotb=plotb)
    if control_switch_list[7]:
        density_of_state(atomo,atomb,atoma,atomc)
    
    
control_switch_list = [0,0,0,'#',0,0,0,0]    
    
if By == Cy:
    #pseudo code:
    #if L1 == L2 => this is still tilt diamond 
    #if not => this is hexagon(vertical hexagon) 
    if round(distance.euclidean(atomAp, atomO),10) == round(distance.euclidean(atomB, atomO),10):
        #the supercell is tilt diamond shape as before
        print("supercell is 'horizontal diamond' ")
        can = count_atom_num(atomO,atomB,atomAp,atomC)
        operate_main_fn(atomO,atomB,atomAp,atomC, control_switch_list)
        
    else:
        if round(distance.euclidean(atomAp, atomB),10) == round(distance.euclidean(atomO, atomB),10):
            print("supercell is 'vertical diamond' ")
            can = count_atom_num(atomO,atomB,atomAp,atomC)
            operate_main_fn(atomO,atomB,atomAp,atomC, control_switch_list)

        else:
            print("supercell is 'vertical hexagon' ")
            if By > Apy:
                #type 1 vertical hexagon A' is lower than B
                Ay = test3[np.where(test3[:,1]>Apy)][:,1].min()
                Ax = Apx
                atomA = [Ax,Ay]

                can = count_atom_num(atomO,atomB,atomA,atomC)
                operate_main_fn(atomO,atomB,atomA,atomC, control_switch_list)
                
            else:
                #type 2 vertical hexagon A' is higher than B
                Bpx = test2[np.where(test2[:,0]==Bx)][:,0].min()
                Bpy = test2[np.where(test2[:,1]>By)][:,1].min()
                atomBp = [Bpx,Bpy]
                
                Cpx = test1[np.where(test1[:,0]==Cx)][:,0].min()
                Cpy = test1[np.where(test1[:,1]>Cy)][:,1].min()
                atomCp = [Cpx,Cpy]
                
                Ay = test3[np.where(test3[:,1]>Apy)][:,1].min()
                Ax = Apx
                atomA = [Ax,Ay]

                can = count_atom_num(atomO,atomBp,atomA,atomCp)
                operate_main_fn(atomO,atomBp,atomA,atomCp, control_switch_list)
    
else:
    #the supercell is hexagon shape, this hexagon is horizontal hexagon   
    print("supercell is 'horizontal hexagon' ")
    if Apy > By:
        #type 1 horizontal hexagon A' is higher than B
        Bpx = test2[np.where(test2[:,0]>Bx)][:,0].min()
        Bpy = test2[np.where(test2[:,1]==By)][:,1].max()
        atomBp = [Bpx,Bpy]
        
        Cpx = test1[np.where(test1[:,0]<Cx)][:,0].max()
        Cpy = test1[np.where(test1[:,1]<Cy)][:,1].max()
        atomCp = [Cpx,Cpy]

        can = count_atom_num(atomO,atomBp,atomAp,atomCp)
        operate_main_fn(atomO,atomBp,atomAp,atomCp, control_switch_list)
        
    else:
        #type 2 horizontal hexagon A' is equal high with B
        Bpx = test2[np.where(test2[:,0]>Bx)][:,0].min()
        Bpy = test2[np.where(test2[:,1]<By)][:,1].max()
        atomBp = [Bpx,Bpy]
        
        Cpx = test1[np.where(test1[:,0]<Cx)][:,0].max()
        Cpy = Cy #in this case, Cpy = Cy = Bpy, so they are equal high
        atomCp = [Cpx,Cpy]

        can = count_atom_num(atomO,atomBp,atomAp,atomCp)   
        operate_main_fn(atomO,atomBp,atomAp,atomCp, control_switch_list)
              
###############################################################################


end1 = time.time() #結束畫圖的時間
print(f"Total execution time：{round(end1 - time_1,8)} 秒")
plt.show()


###############################################################################
