#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:06:31 2022

@author: david
"""





#------------------------------------------------------------------------------
#------------------PROJECT NLA 1. DAVID ROSADO---------------------------------
#------------------------------------------------------------------------------
#------------------General case, second dataset---------------------------------

import numpy as np
import time
from scipy.linalg import ldl,solve_triangular

#Function to read an external matrix
def read_matrix(path,n,m,symmetric = False):
    with open(path, "r") as file:
        mat=file.readlines()
    matrix=np.zeros((n,m))
    for line in mat:
        row, column, val=line.strip().split()
        matrix[int(row)-1,int(column)-1]=float(val)
        if symmetric == True:
            matrix[int(column)-1, int(row)-1]=float(val)
    return matrix
#Function to read an external vector
def read_vector(path,n):
    with open(path, "r") as file:
        vect=file.readlines()
    vector=np.zeros(n)
    for line in vect:
        ind,val=line.strip().split()
        vector[int(ind)-1]=float(val)
    return vector
#Function that creates the KKT matrix
def KKT(A,G,C,m,n,p,z):
    N=n + p + 2*m
    M_KKT=np.zeros((N,N))
    
    #First row
    for i in range(0,n):
        for j in range(0,n):
            M_KKT[i,j]=G[i,j]
        for j in range(n,n+p):
            M_KKT[i,j]=-A[i,j-n]
        for j in range(n+p,n+p+m):
            M_KKT[i,j]=-C[i,j-n-p]
    #First column        
    for j in range(0,n):
        for i in range(n,n+p):
            M_KKT[i,j]=-A[j,i-n]
        for i in range(n+p,n+p+m):
            M_KKT[i,j]=-C[j,i-n-p]
    #Bottom right matrix
    for i in range(m+n+p,N):
        M_KKT[i,i-m]=z[i]
        M_KKT[i-m,i]=1
        M_KKT[i,i]=z[i-m]
    
    return M_KKT
#Function that computes the function F
def F(A,G,C,d,g,b,n,m,p,z):
    N=n + p + 2*m
    #Initializations of variables
    
    F=np.zeros(N)
    x=np.zeros(n)
    gamma=np.zeros(p)
    lanbda=np.zeros(m)
    s=np.zeros(m)
    
    for i in range(0,n):
        x[i]=z[i]
    for i in range(0,p):
        gamma[i]=z[i+n]
    for i in range(0,m):
        lanbda[i]=z[i+n+p]
        s[i]=z[i+n+p+m]
    
    #First component of F
    aux=np.zeros(n)
    aux=np.matmul(G,x) + g -np.matmul(A,gamma)- np.matmul(C,lanbda)
    for i in range(0,n):
        F[i]=aux[i]
    #Second component of F
    aux=np.zeros(p)
    aux=np.matmul(A.T,x)
    for i in range(n,n+p):
        F[i]=b[i-n]-aux[i-n]
    #Third component of F
    aux=np.zeros(m)
    aux=s+d-np.matmul(C.T,x)
    for i in range(n+p,n+p+m):
        F[i]=aux[i-n-p]
    #Last component of F
    aux=s*lanbda
    for i in range(n+p+m,N):
        F[i]=aux[i-n-m-p]
    
    return F

def Newton_step(lamb0,dlamb,s0,ds):
    alp=1
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    
    return alp

#Function that computes the Newton modified algorithm using np.linalg.solve
def Newton_mod(A,G,C,d,g,b,n,m,p,z0):
    #Varaibles we need
    N=n+p+2*m
    lanbda=np.zeros(m)
    s=np.zeros(m)
    delta_lanbda=np.zeros(m)
    delta_s=np.zeros(m)
    e=np.ones(m)
    new_vect=np.zeros(N)
    #Conditions to enter in the while
    iterations=0
    r_L=np.ones(n)
    r_A=np.ones(p)
    r_C=np.ones(m)
    mu=1
    while(np.linalg.norm(r_L,2)>1e-16 and np.linalg.norm(r_A,2)>1e-16 and np.linalg.norm(r_C,2)>1e-16 and abs(mu)>1e-16 and iterations<=100):
       
        #1.Predictor substep 
        
        M_KKT=KKT(A,G,C,m,n,p,z0)
        delta_z=np.linalg.solve(M_KKT,-F(A,G,C,d,g,b,n,m,p,z0))
        
        #2.Step-size correction substep
        
        for i in range(0,m):
            lanbda[i]=z0[i+n+p]
            s[i]=z0[i+n+p+m]
        for i in range(n+p,n+m+p):
            delta_lanbda[i-n-p]=delta_z[i]
            delta_s[i-n-p]=delta_z[i+m]
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #3.Several computations
        
        mu=np.dot(s,lanbda)/m
        mu_tilde=np.dot(s+alpha*delta_s,lanbda+alpha*delta_lanbda)/m
        sigma=(mu_tilde/mu)**3
        
        #4.Corrector substep
        
        new_vect=-F(A,G,C,d,g,b,n,m,p,z0)
        new_vect[n+p+m:N]=new_vect[n+p+m:N] -delta_s*delta_lanbda +sigma*mu*e
        delta_z=np.linalg.solve(M_KKT,new_vect)
        
        #5.Step-size correction substep
        
        for i in range(n+p,n+m+p):
            delta_lanbda[i-n-p]=delta_z[i]
            delta_s[i-n-p]=delta_z[i+m]
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #6.Update substep
        
        z0=z0+0.95*alpha*delta_z
        iterations=iterations+1
        r_L=F(A,G,C,d,g,b,n,m,p,z0)[0:n]
        r_A=F(A,G,C,d,g,b,n,m,p,z0)[n:n+p]
        r_C=F(A,G,C,d,g,b,n,m,p,z0)[n+p:n+p+m]
        

    return z0,iterations
    
#Function that computes the Newton modifed algorithm with strategy 1
def Newton_mod_strat1(A,G,C,d,g,b,n,m,p,z0):
    N=n+p+2*m
    MAT=np.zeros((n+m+p,n+m+p))
    VECT=np.zeros(n+m+p)
    lanbda=np.zeros(m)
    s=np.zeros(m)
    delta_z=np.zeros(N)
    delta_lanbda=np.zeros(m)
    delta_s=np.zeros(m)
    e=np.ones(m)
    new_vect=np.zeros(N)
    z=np.zeros(n+m+p)
    Blk=np.zeros((2,2))
    Blk_v=np.zeros(2)
    #Conditions to enter in the while
    iterations=0
    r_L=np.ones(n)
    r_A=np.ones(p)
    r_C=np.ones(m)
    r_s=np.ones(m)
    mu=1
    #Let us start creating the 3x3 block matrix. Notice that only one block is changing
    #First row
    for i in range(0,n):
        for j in range(0,n):
            MAT[i,j]=G[i,j]
        for j in range(n,n+p):
            MAT[i,j]=-A[i,j-n]
        for j in range(n+p,n+p+m):
            MAT[i,j]=-C[i,j-n-p]
    #First column
    for j in range(0,n):
        for i in range(n,n+p):
            MAT[i,j]=-A[j,i-n]
        for i in range(n+p,n+p+m):
            MAT[i,j]=-C[j,i-n-p]
    
    while(np.linalg.norm(r_L,2)>1e-16 and np.linalg.norm(r_A,2)>1e-16 and np.linalg.norm(r_C,2)>1e-16 and abs(mu)>1e-16 and iterations<=100):
        #Initializations of variables
        for i in range(0,m):
            lanbda[i]=z0[i+n+p]
            s[i]=z0[i+n+p+m]
        r_L=F(A,G,C,d,g,b,n,m,p,z0)[0:n]
        r_A=F(A,G,C,d,g,b,n,m,p,z0)[n:n+p]
        r_C=F(A,G,C,d,g,b,n,m,p,z0)[n+p:n+p+m]
        r_s=F(A,G,C,d,g,b,n,m,p,z0)[n+p+m:N]
        
        #Let us update the new block matrix 2x2, only the last block is modified
        for i in range(n+p,n+p+m):
            MAT[i,i]=-s[i-n-p]/lanbda[i-n-p]
        #Let us create the vector b to solve the system Ax=b, with A=MAT
        VECT[0:n]=-r_L
        VECT[n:n+p]=-r_A
        VECT[n+p:n+p+m]=-r_C + r_s/lanbda
        
        #1.Predictor substep 
        #ldl^t factorization 
        L,D,perm=ldl(MAT)
        #Now the permutation matrix is not the identity
        #See the memory for detailed information
        #First we solve Lx=VECT with the corresponding permutations
        #This allows us to use solve_triangular
        LL=L[perm,:]
        y=solve_triangular(LL,VECT[perm],lower=True,unit_diagonal=True)
        #D is diagonal per blocks
        #We know that that D[0,0] and D[-1,-1] are different from 0
        z[0]=y[0]/D[0,0]
        z[-1]=y[-1]/D[-1,-1]
        for i in range(1,n+m+p-1):
            if D[i,i-1]!=0:
                Blk[0,0]=D[i-1,i-1]
                Blk[0,1]=D[i-1,i]
                Blk[1,0]=D[i,i-1]
                Blk[1,1]=D[i,i]
                Blk_v[0]=y[i-1]
                Blk_v[1]=y[i]
                v=np.linalg.solve(Blk,Blk_v)
                z[i-1]=v[0]
                z[i]=v[1]
                i=i+1
            else:
                if D[i,i]!=0:
                    z[i]=y[i]/D[i,i]
        #Final we solve the last system, L^tx=z, taking advantage of the structure             
        delta_solve=solve_triangular(LL.T,z,lower=False,unit_diagonal=True)
        P=np.identity(n+m+p)
        P=P[perm,:]
        delta_solve=np.matmul(P.T,delta_solve)
        
        #Construction of delta_z
        for i in range(n+p,m+n+p):
            delta_lanbda[i-n-p]=delta_solve[i]
        delta_s=(-r_s-s*delta_lanbda)/lanbda
        for i in range(0,n+p):
            delta_z[i]=delta_solve[i]
        for i in range(n+p,n+m+p):
            delta_z[i]=delta_lanbda[i-n-p]
            delta_z[i+m]=delta_s[i-n-p]
            
        #2.Step-size correction substep
        
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #3.Several computations
        
        mu=np.dot(s,lanbda)/m
        mu_tilde=np.dot(s+alpha*delta_s,lanbda+alpha*delta_lanbda)/m
        sigma=(mu_tilde/mu)**3
        
        #4.Corrector substep
        new_vect=F(A,G,C,d,g,b,n,m,p,z0)
        new_vect[n+m+p:N]=new_vect[n+m+p:N] - sigma*mu*e +delta_s*delta_lanbda
        r_s=new_vect[n+m+p:N]
        #First we solve Lx=new_vect with the corresponding permutations
        #This allows us to use solve_triangular
        y=solve_triangular(LL,np.concatenate((-r_L,-r_A, -r_C + r_s/lanbda),axis=0)[perm],lower=True,unit_diagonal=True)
        #D is diagonal per blocks
        #We know that that D[0,0] and D[-1,-1] are different from 0
        z[0]=y[0]/D[0,0]
        z[-1]=y[-1]/D[-1,-1]
        for i in range(1,n+m+p-1):
            if D[i,i-1]!=0:
                Blk[0,0]=D[i-1,i-1]
                Blk[0,1]=D[i-1,i]
                Blk[1,0]=D[i,i-1]
                Blk[1,1]=D[i,i]
                Blk_v[0]=y[i-1]
                Blk_v[1]=y[i]
                v=np.linalg.solve(Blk,Blk_v)
                z[i-1]=v[0]
                z[i]=v[1]
                i=i+1
            else:
                if D[i,i]!=0:
                    z[i]=y[i]/D[i,i]
        #Final we solve the last system, L^tx=z, taking advantage of the structure             
        delta_solve=solve_triangular(LL.T,z,lower=False,unit_diagonal=True)
        delta_solve=np.matmul(P.T,delta_solve)
        #Actualization of delta_lanbda and delta_s
        for i in range(n+p,m+n+p):
            delta_lanbda[i-n-p]=delta_solve[i]
        delta_s=(-r_s-s*delta_lanbda)/lanbda
        
        #5.Step-size correction substep
       
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        #Construction of delta_z
        for i in range(0,n+p):
            delta_z[i]=delta_solve[i]
        for i in range(n+p,n+m+p):
            delta_z[i]=delta_lanbda[i-n-p]
            delta_z[i+m]=delta_s[i-n-p]
        
        #6.Update substep
        z0=z0+0.95*alpha*delta_z
        iterations=iterations+1
    return z0,iterations 




#------------------------------------------------------------------------------
#-----------------------------MAIN---------------------------------------------
#------------------------------------------------------------------------------
#read the matrices
n=1000
p=500
m=2000
N=n+p+2*m
A=read_matrix('A.dad',n,p)
C=read_matrix('C.dad',n,m)
G=read_matrix('G.dad',n,n,True)
g=read_vector('g.dad',n)
b=read_vector('b.dad',p)
d=read_vector('d.dad',m) 
#Initial condition
z0=np.zeros(N)
for i in range(n,N):
    z0[i]=1
    
#Method using np.linalg.solve
st0=time.time()    
res0,it0=Newton_mod(A, G, C, d, g, b, n, m, p, z0)
et0=time.time()
if it0<100:
    print("METHOD USING np.linalg.norm ACHIEVED")
    print('Newtons method finish with',it0,'iterations')
    #Value of f(x)
    val=(1/2)*np.dot(res0[0:n],np.matmul(G,res0[0:n])) + np.dot(g,res0[0:n])
    print('The solution is a vector x such that f(x)=',val)
    print('The execution time is',et0-st0,'seconds')
else:
    print('Maximum iterations allowed for the Newtons method using np.linalg.solve')
#Method using ldl^t factorization
st1=time.time()    
res1,it1=Newton_mod_strat1(A, G, C, d, g, b, n, m, p, z0)
et1=time.time()
if it1<100:
    print("METHOD USING ldl^t ACHIEVED")
    print('Newtons method finish with',it1,'iterations')
    #Value of f(x)
    val=(1/2)*np.dot(res1[0:n],np.matmul(G,res1[0:n])) + np.dot(g,res1[0:n])
    print('The solution is a vector x such that f(x)=',val)
    print('The execution time is',et1-st1,'seconds')
else:
    print('Maximum iterations allowed for the Newtons method using ldl^t factorization')