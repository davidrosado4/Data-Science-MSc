#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:16:52 2022

@author: david
"""

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import time

#------------------------------------------------------------------------------
#------------------------------PROJECT 3---------------------------------------
#------------------------NUMERICAL LINEAR ALGEBRA------------------------------
#------------------------David Rosado Rodríguez--------------------------------
#------------------------------------------------------------------------------
#We will work with sparse matrices!!
#Remember: The output of a sparse matrix is of the following form:
    #(i,j) number if number!=0


#Function that creates the sparse matrix D
def create_D(G):
    #Let us compute the out degree of the page j
    n_j = np.sum(G,axis=0)
    #Let us compute the diagonal of the desire matrix
    d = np.zeros(G.shape[0])
    for i in range(0,G.shape[0]):
        if n_j[0,i] == 0:
            d[i] = 0
        else:
            d[i]=1/n_j[0,i]
    #Return a matrix which diagonal is d
    #We return D as a sparse matrix. 
    return sp.diags(d)
#Funcion that creates the sparse matrix A=GD,
def create_A(D,G):
    A = G.dot(D)
    return A
#Function that computes the PR vector of M_m using the power method (storing matrices)
def PR_store(A,tol,m):
    n = A.shape[0]
    
    #Let us initialize some vectors in order to start the iterative algorithm 
    e = np.ones(n)
    z = np.ones(n)/n
    #A.indices gives the column position of the non-zero values
    z[np.unique(A.indices)] = m/n
    x_0 = np.zeros(n)
    x_k = np.ones(n) / n 
    
    #Iterative algorithm
    while np.linalg.norm(x_0-x_k,np.inf)>tol:
        x_0 = x_k
        x_k = (1-m)*A.dot(x_0) + e*(np.dot(z,x_0))

    #Normalization of the vector
    x_k = x_k / np.sum(x_k)
    return x_k
#Function that computes the PR vector of M_m using the power method ( without storing matrices)
def PR_without_store(G,tol,m):
    n = G.shape[0]
    #Let us compute the vector L and n_j
    L = []
    n_j = []
    for j in range(0,n):
        #webpages with link to page j
        L_j = G.indices[G.indptr[j]:G.indptr[j+1]]
        L.append(L_j)
        #n_j = length of L_j
        n_j.append(len(L_j))
    
    #Initialize some vectors in order to start the iterative algorithm
    x = np.zeros(n)
    xc = np.ones(n) / n 
    #Given code!
    while np.linalg.norm(x-xc,np.inf)>tol:
        xc = x
        x = np.zeros(n)
        for j in range (0, n):
            if(n_j[j] == 0):
                x = x + xc[j] / n
            else:
                for i in L[j]:
                    x[i] = x[i] + xc[j] / n_j[j]
        x = (1 - m) * x + m / n
    #Normalization of the vector
    x = x / np.sum(x)
    return x
        
if __name__ == '__main__':
    
    #Let us read the matrix G, Gnutella30.mtx
    G = mmread('p2p-Gnutella30.mtx')
    #Let us create the matrix A
    D = create_D(G)
    A = create_A(D,G)
    #Let us solve compute the PR vector of M_m using the power method (storing matrices)
    #We will also compute the computational time
    st1 = time.time()
    x1 = PR_store(A,1e-12,0.15)
    et1 = time.time()
    print('The solution of the PR vector of M_m using the power method (storing matrices) is')
    print(x1)
    print('Computational time:', et1-st1,'seconds')
    print('\n')
    #Let us solve compute the PR vector of M_m using the power method ( without storing matrices)
    #We will also compute the computational time
    st2 = time.time()
    x2 = PR_without_store(sp.csc_matrix(G),1e-12,0.15)
    et2 = time.time()
    print('The solution of the PR vector of M_m using the power method (without storing matrices) is')
    print(x2)
    print('Computational time:', et2-st2,'seconds')
    print('\n')
    print('The difference between both solution is',np.linalg.norm(x1-x2))
    print('\n')
    
    
    
    
    
  