#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:27:42 2022

@author: david
"""



#------------------------------------------------------------------------------
#------------------PROJECT NLA 1. DAVID ROSADO---------------------------------
#------------------------------------------------------------------------------
#------------First case, inequality constrains case (i.e. A=0)-----------------


import numpy as np
import time
from scipy.linalg import ldl, cholesky,solve_triangular
import matplotlib.pyplot as plt

#Function that creats the KKT matrix
def KKT(G,C,n,z):
    m=2*n
    N=n + 2*m
    M_KKT=np.zeros((N,N))
    
    #First row
    for i in range(0,n):
        for j in range(0,n):
            M_KKT[i,j]=G[i,j]
        for j in range(n,n+m):
            M_KKT[i,j]=-C[i,j-n]
            
    #First column
    for j in range(0,n):
        for i in range(n,n+m):
            M_KKT[i,j]=-C[j,i-n]
            
    #Bottom right matrix
    for i in range(n+m,N):
        M_KKT[i,i-m]=z[i]
        M_KKT[i-m,i]=1
        M_KKT[i,i]=z[i-m]
        
    return M_KKT
#Function that computes the function F
def F(G,C,d,g,n,z):
    m=2*n
    N=n + 2*m
    #Initializations of variables
    
    F=np.zeros(N)
    x=np.zeros(n)
    lanbda=np.zeros(m)
    s=np.zeros(m)
    
    for i in range(0,n):
        x[i]=z[i]
    for i in range(0,m):
        lanbda[i]=z[i+n]
        s[i]=z[i+n+m]
    
    #First component of F
    aux=np.zeros(n)
    aux=np.matmul(G,x) + g - np.matmul(C,lanbda)
    for i in range(0,n):
        F[i]=aux[i]
    #Second component of F
    aux=np.zeros(m)
    aux=s+d-np.matmul(C.T,x)
    for i in range(n,n+m):
        F[i]=aux[i-n]
    #Last component of F
    aux=s*lanbda
    for i in range(n+m,N):
        F[i]=aux[i-n-m]
    
    return F
#Function that computes the step-size correction substep(given code)
#PROBLEM C1
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
#PROBLEM C2,C3
def Newton_mod(G,C,d,g,n,z0,boolean=False):
    #Varaibles we need
    m=2*n
    N=n+2*m
    lanbda=np.zeros(m)
    s=np.zeros(m)
    delta_lanbda=np.zeros(m)
    delta_s=np.zeros(m)
    e=np.ones(m)
    new_vect=np.zeros(N)
    condnumber=np.zeros(100)
    #Conditions to enter in the while
    iterations=0
    r_L=np.ones(n)
    r_C=np.ones(m)
    mu=1
    while(np.linalg.norm(r_L,2)>1e-16 and np.linalg.norm(r_C,2)>1e-16 and abs(mu)>1e-16 and iterations<=100):
        #1.Predictor substep 
        
        M_KKT=KKT(G,C,n,z0)
        delta_z=np.linalg.solve(M_KKT,-F(G,C,d,g,n,z0))
        #We compute the condition number of the matrix for every step
        if(boolean==True):
            condnumber[iterations]=np.linalg.cond(M_KKT,2)
        
        #2.Step-size correction substep
        
        for i in range(0,m):
            lanbda[i]=z0[i+n]
            s[i]=z0[i+n+m]
        for i in range(n,n+m):
            delta_lanbda[i-n]=delta_z[i]
            delta_s[i-n]=delta_z[i+m]
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #3.Several computations
        
        mu=np.dot(s,lanbda)/m
        mu_tilde=np.dot(s+alpha*delta_s,lanbda+alpha*delta_lanbda)/m
        sigma=(mu_tilde/mu)**3
        
        #4.Corrector substep
        new_vect=-F(G,C,d,g,n,z0)
        new_vect[n+m:N]=new_vect[n+m:N] -delta_s*delta_lanbda + sigma*mu*e
     
        delta_z=np.linalg.solve(M_KKT,new_vect)
        
        #5.Step-size correction substep
        
        for i in range(n,n+m):
            delta_lanbda[i-n]=delta_z[i]
            delta_s[i-n]=delta_z[i+m]
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #6.Update substep
        
        z0=z0+0.95*alpha*delta_z
        iterations=iterations+1
        r_L=F(G,C,d,g,n,z0)[0:n]
        r_C=F(G,C,d,g,n,z0)[n:n+m]
    return z0,iterations,condnumber
#Function that computes the Newton modifed algorithm with strategy 1
#PROBLEM C4
def Newton_mod_strat1(G,C,d,g,n,z0,boolean=False):
    #Varaibles we need
    m=2*n
    N=n+2*m
    MAT=np.zeros((n+m,n+m))
    VECT=np.zeros(n+m)
    lanbda=np.zeros(m)
    s=np.zeros(m)
    delta_z=np.zeros(N)
    delta_solve=np.zeros(n+m)
    delta_lanbda=np.zeros(m)
    delta_s=np.zeros(m)
    z=np.zeros(n+m)
    e=np.ones(m)
    new_vect=np.zeros(N)
    condnumber=np.zeros(100)
    #Conditions to enter in the while
    iterations=0
    r_1=np.ones(n)
    r_2=np.ones(m)
    r_3=np.ones(m)
    mu=1
    #Let us start creating the matrix to apply ldl^t
    #Notice that the matrix is 2x2 block matrix and only the last block is changing
    #First row
    for i in range(0,n):
        for j in range(0,n):
            MAT[i,j]=G[i,j]
        for j in range(n,m+n):
            MAT[i,j]=-C[i,j-n]
     #First column   
    for j in range(0,n):
        for i in range(n,n+m):
            MAT[i,j]=-C[j,i-n]
    
    while(np.linalg.norm(r_1,2)>1e-16 and np.linalg.norm(r_2,2)>1e-16 and abs(mu)>1e-16 and iterations<=100):
        #Initializations of variables

        for i in range(0,m):
            lanbda[i]=z0[i+n]
            s[i]=z0[i+n+m]
        r_1=F(G,C,d,g,n,z0)[0:n]
        r_2=F(G,C,d,g,n,z0)[n:n+m]
        r_3=F(G,C,d,g,n,z0)[n+m:N]

        #Let us update the new block matrix 2x2, only the last block is modified
        for i in range(n,n+m):
            MAT[i,i]=-s[i-n]/lanbda[i-n]
        #Let us create the vector b to solve the system Ax=b, with A=MAT
        VECT[0:n]=-r_1
        VECT[n:n+m]=-r_2+r_3/lanbda
        #We compute the condition number of the matrix for every step
        if(boolean==True):
            condnumber[iterations]=np.linalg.cond(MAT,2)
        
        #1.Predictor substep 
        #ldl^t factorization 
        
        L,D,perm=ldl(MAT)
        for i in range(0,n+m):
            if perm[i]!=i:
                print('The permutation is no the identity')
                exit()
        #The permutation matrix is the identity
        y=solve_triangular(L,VECT,lower=True,unit_diagonal=True)
        for i in range(0,n+m):
            z[i]=y[i]/D[i,i]
        delta_solve=solve_triangular(L.T,z,lower=False,unit_diagonal=True)
        
        #Construction of delta_z
        for i in range(n,m+n):
            delta_lanbda[i-n]=delta_solve[i]
        delta_s=(-r_3-s*delta_lanbda)/lanbda
        for i in range(0,n):
            delta_z[i]=delta_solve[i]
        for i in range(n,n+m):
            delta_z[i]=delta_lanbda[i-n]
            delta_z[i+m]=delta_s[i-n]
        
        #2.Step-size correction substep
        
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        
        #3.Several computations
        
        mu=np.dot(s,lanbda)/m
        mu_tilde=np.dot(s+alpha*delta_s,lanbda+alpha*delta_lanbda)/m
        sigma=(mu_tilde/mu)**3
        
        #4.Corrector substep
        new_vect=F(G,C,d,g,n,z0)
        new_vect[n+m:N]=new_vect[n+m:N] - sigma*mu*e +delta_s*delta_lanbda
        r_3=new_vect[n+m:N]
        
        y=solve_triangular(L,np.concatenate((-r_1,-r_2 + r_3/lanbda),axis=0),lower=True,unit_diagonal=True)
        for i in range(0,n+m):
            z[i]=y[i]/D[i,i]
        delta_solve=solve_triangular(L.T,z,lower=False,unit_diagonal=True)
        #Actualization of delta_lanbda and delta_s
        for i in range(n,m+n):
            delta_lanbda[i-n]=delta_solve[i]
        delta_s=(-r_3-s*delta_lanbda)/lanbda
        
        #5.Step-size correction substep
       
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
        #Construction of delta_z
        for i in range(0,n):
            delta_z[i]=delta_solve[i]
        for i in range(n,n+m):
            delta_z[i]=delta_lanbda[i-n]
            delta_z[i+m]=delta_s[i-n]
        
        #6.Update substep
        z0=z0+0.95*alpha*delta_z
        iterations=iterations+1
    return z0,iterations,condnumber
#Function that computes the Newton modifed algorithm with strategy 1
#PROBLEM C4
def Newton_mod_strat2(G,C,d,g,n,z0,boolean=False):
    #Variables we need
    m=2*n
    N=n+2*m
    lanbda=np.zeros(m)
    s=np.zeros(m)
    delta_z=np.zeros(N)
    delta_x=np.zeros(n)
    delta_lanbda=np.zeros(m)
    G_hat=np.zeros((n,n))
    r_hat=np.zeros(n)
    new_vect=np.zeros(N)
    e=np.ones(m)
    condnumber=np.zeros(100)
    #Conditions to enter in the while
    iterations=0
    r_1=np.ones(n)
    r_2=np.ones(m)
    r_3=np.ones(m)
    mu=1
    while(np.linalg.norm(r_1,2)>1e-16 and np.linalg.norm(r_2,2)>1e-16 and abs(mu)>1e-16 and iterations<=100):
        #Initializations of variables

        for i in range(0,m):
            lanbda[i]=z0[i+n]
            s[i]=z0[i+n+m]
        r_1=F(G,C,d,g,n,z0)[0:n]
        r_2=F(G,C,d,g,n,z0)[n:n+m]
        r_3=F(G,C,d,g,n,z0)[n+m:N]
        #Let us create the matrix G_hat and the r_hat vector
        G_hat=G+np.matmul(C/s*lanbda,C.T)
        r_hat=np.matmul(-C/s,-r_3+lanbda*r_2)
        
        #1.Predictor substep
        #cholesky factorization
        
        #We compute the condition number of the matrix for every step
        if(boolean==True):
            condnumber[iterations]=np.linalg.cond(G_hat,2)
        L=cholesky(G_hat)
        y=solve_triangular(L,-r_1-r_hat,lower=True,unit_diagonal=False)
        delta_x=solve_triangular(L.T,y,lower=False,unit_diagonal=False)
        #Construction of delta_z
        delta_lanbda=(-r_3+lanbda*r_2)/s - (lanbda/s)*np.matmul(C.T,delta_x)
        delta_s=-r_2+np.matmul(C.T,delta_x)
        for i in range(0,n):
            delta_z[i]=delta_x[i]
        for i in range(n,n+m):
            delta_z[i]=delta_lanbda[i-n]
            delta_z[i+m]=delta_s[i-n]
            
        #2.Step-size correction substep
         
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)
         
        #3.Several computations
         
        mu=np.dot(s,lanbda)/m
        mu_tilde=np.dot(s+alpha*delta_s,lanbda+alpha*delta_lanbda)/m
        sigma=(mu_tilde/mu)**3
         
        #4.Corrector substep
        new_vect=F(G,C,d,g,n,z0) 
        new_vect[n+m:N]=new_vect[n+m:N] - sigma*mu*e +delta_s*delta_lanbda
        r_3=new_vect[n+m:N]
        #actualize r_hat and solve the system with cholesky factorization
        r_hat=np.matmul(-C/s,-r_3+lanbda*r_2)
        y=solve_triangular(L,-r_1-r_hat,lower=True,unit_diagonal=False)
        delta_x=solve_triangular(L.T,y,lower=False,unit_diagonal=False)
        #Construction of delta_z
        delta_lanbda=(-r_3+lanbda*r_2)/s-np.matmul(C.T,delta_x)*lanbda/s
        delta_s=-r_2+np.matmul(C.T,delta_x)
        for i in range(0,n):
            delta_z[i]=delta_x[i]
        for i in range(n,n+m):
            delta_z[i]=delta_lanbda[i-n]
            delta_z[i+m]=delta_s[i-n]
            
        #5.Step-size correction substep
       
        alpha=Newton_step(lanbda,delta_lanbda,s,delta_s)

        #6.Update substep
        z0=z0+0.95*alpha*delta_z
        iterations=iterations+1
    return z0,iterations,condnumber
#------------------------------------------------------------------------------
#-----------------------------MAIN---------------------------------------------
#------------------------------------------------------------------------------      
#We will do the program for different dimensions n
f0 = open("C2,C3-Problem.txt", "w")
f1 = open("C4-Problem_strat1.txt","w")
f2 = open("C4-Problem_strat2.txt","w")
f0.write("C2,C3-PROBLEM\n\n")
f1.write("C4-PROBLEM_strat1\n\n")
f2.write("C4-PROBLEM_strat2\n\n")
#We create three arrays for the plot of executotion time/ dimension n
dimension=np.zeros(97)
execut0=np.zeros(97)
execut1=np.zeros(97)
execut2=np.zeros(97)
#We create three arrays more to save the number of iterations at every n
itera0=np.zeros(97)
itera1=np.zeros(97)
itera2=np.zeros(97)
#We create the arrays more to save the precision
prec0=np.zeros(97)
prec1=np.zeros(97)
prec2=np.zeros(97)
for n in range(3,100):
    dimension[n-3]=n
    f0.write("case n=")
    f0.write(str(n))
    f0.write("\n")
    f1.write("case n=")
    f1.write(str(n))
    f1.write("\n")
    f2.write("case n=")
    f2.write(str(n))
    f2.write("\n")
    print('------------------------CASE n=',n,'------------------------------')
    print('------------------------------------------------------------------')
    m=2*n
    N=n+2*m
    #Initialization of the matrices and vectors
    G=np.identity(n)
    C=np.zeros((n,m))
    d=np.zeros(m)
    g=np.zeros(n)
    for i in range(0,n):
        C[i,i]=1
        C[i,i+n]=-1
        g[i]=np.random.normal(0,1)
    for i in range(0,m):
        d[i]=-10
    z0=np.zeros(N)
    for i in range(n,N):
        z0[i]=1
        
        
    #PROBLEM C2-C3, using np.linalg.solve
    #Get the start time
    st0=time.time()
    res0,it0,condnumber0=Newton_mod(G, C, d, g, n, z0)
    #Get the end time
    et0=time.time()
    execut0[n-3]=et0-st0
    itera0[n-3]=it0
    prec0[n-3]=np.linalg.norm(res0[0:n]+g,2)
    
    #PROBLEM C4, using strategy 1
    #Get the start time
    st1=time.time()
    res1,it1,condnumber1=Newton_mod_strat1(G, C, d, g, n, z0)
    #Get the end time
    et1=time.time()
    execut1[n-3]=et1-st1
    itera1[n-3]=it1
    prec1[n-3]=np.linalg.norm(res1[0:n]+g,2)
    
    #PROBLEM C4, using strategy 2
    #Get the start time
    st2=time.time()
    res2,it2,condnumber2=Newton_mod_strat2(G, C, d, g, n, z0)
    #Get the end time
    et2=time.time()
    execut2[n-3]=et2-st2
    itera2[n-3]=it2
    prec2[n-3]=np.linalg.norm(res2[0:n]+g,2)
    
    
    #Print the solutions
    #We print in the files the execution time for every n
    #We print in the screen the results for every n
    if it0<100:
        f0.write("Execution time:")
        f0.write(str(execut0[n-3]))
        f0.write("seconds\n")
        f0.write("Iterations:")
        f0.write(str(itera0[n-3]))
        f0.write("\n")
        f0.write("Precision:")
        f0.write(str(prec0[n-3]))
        f0.write("\n")
        if prec0[n-3]<1e-10:
            print('The solution for the inequality constrains case using np.linalg.solve is -g')
        else:
            print('The solution for the inequality constrains case using np.linalg.solve is not correct')
            print(res0[0:n])
            print(-g)
    else:
        print('Maximum iterations allowed for n=',n,'in the case np.linalg.solve')
        f0.write('Maximum iterations allowed for n=')
        f0.write(str(n))
    if it1<100:
        f1.write("Execution time:")
        f1.write(str(execut1[n-3]))
        f1.write("seconds\n")
        f1.write("Iterations:")
        f1.write(str(itera1[n-3]))
        f1.write("\n")
        f1.write("Precision:")
        f1.write(str(prec1[n-3]))
        f1.write("\n")
        if prec1[n-3]<1e-10:
            print('The solution for the inequality constrains case using strategy 1 is -g')
        else:
            print('The solution for the inequality constrains case using strategy 1 is not correct')
            print(res1[0:n])
            print(-g)
    else:
        print('Maximum iterations allowed for n=',n,'in the strategy 1 case')
        f1.write('Maximum iterations allowed for n=')
        f1.write(str(n)) 
    if it2<100:
        f2.write("Execution time:")
        f2.write(str(execut2[n-3]))
        f2.write("seconds\n")
        f2.write("Iterations:")
        f2.write(str(itera2[n-3]))
        f2.write("\n")
        f2.write("Precision:")
        f2.write(str(prec2[n-3]))
        f2.write("\n")
        if prec2[n-3]<1e-10:
            print('The solution for the inequality constrains case using strategy 2 is -g')
        else:
            print('The solution for the inequality constrains case using strategy 2 is not correct')
            print(res2[0:n])
            print(-g)
        print('--------------------------------------------------------------')
    else:
        print('Maximum iterations allowed for n=',n,'in the strategy 2 case')
        f2.write('Maximum iterations allowed for n=')
        f2.write(str(n)) 
#Plot of the executation time/dimension n
plt.figure(1)
plt.plot(dimension,execut0,color='g',label='np.linalg.solve')
plt.plot(dimension,execut1,color='b',label='ldl^t ')
plt.plot(dimension,execut2,color='r',label='cholesky')
plt.title("Execution time")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel("Dimension n")
plt.ylabel("Execution time")


#Plot of number of iterations/dimension n
plt.figure(2)
plt.plot(dimension,itera0,color='g')
plt.title("Number of iterations(np.linalg.solve)")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")
plt.figure(3)
plt.plot(dimension,itera1,color='b')
plt.title("Number of iterations(ldl^t)")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")
plt.figure(4)
plt.plot(dimension,itera2,color='r')
plt.title("Number of iterations(cholesky)")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")


#Plot of the precision/dimension n
plt.figure(5)
plt.plot(dimension,prec0,color='g')
plt.title("Precision(np.linalg.sovle)")
plt.xlabel("Dimension n")
plt.ylabel("Precision")
plt.figure(6)
plt.plot(dimension,prec1,color='b')
plt.title("Precision(ldl^t)")
plt.xlabel("Dimension n")
plt.ylabel("Precision")
plt.figure(7)
plt.plot(dimension,prec2,color='r')
plt.title("Precision(cholesky)")
plt.xlabel("Dimension n")
plt.ylabel("Precision")

#Close the files
f0.close()
f1.close()
f2.close()


#We plot the condition number in every step for some dimensions 
#For example n=20,40,60
#------------------------------------------------------------------------------
#--------------------------------n=20------------------------------------------
#------------------------------------------------------------------------------
n=20
m=2*n
N=n+2*m
#Initialization of the matrices and vectors
G=np.identity(n)
C=np.zeros((n,m))
d=np.zeros(m)
g=np.zeros(n)
for i in range(0,n):
    C[i,i]=1
    C[i,i+n]=-1
    g[i]=np.random.normal(0,1)
for i in range(0,m):
    d[i]=-10
#Let us solve the problem using np.linalg.solve
z0=np.zeros(N)
for i in range(n,N):
    z0[i]=1

res0,it0,condnumber0=Newton_mod(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it0)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it0)
for i in range(0,it0):
    iterations[i]=i
    printcondnumber[i]=condnumber0[i]
plt.figure(8)
plt.plot(iterations,printcondnumber,color='g')
plt.title("Cond.numb(np.linalg.solve), n=20")
plt.xlabel("Iterations")
plt.ylabel("Condition number")


res1,it1,condnumber1=Newton_mod_strat1(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it1)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it1)
for i in range(0,it1):
    iterations[i]=i
    printcondnumber[i]=condnumber1[i]
plt.figure(9)
plt.plot(iterations,printcondnumber,color='b')
plt.title("Cond.numb(ldl^t), n=20")
plt.xlabel("Iterations")
plt.ylabel("Condition number")    


res2,it2,condnumber2=Newton_mod_strat2(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it2)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it2)
for i in range(0,it2):
    iterations[i]=i
    printcondnumber[i]=condnumber2[i]
plt.figure(10)
plt.plot(iterations,printcondnumber,color='r')
plt.title("Cond.numb(cholesky), n=20")
plt.xlabel("Iterations")
plt.ylabel("Condition number")
#------------------------------------------------------------------------------
#--------------------------------n=40------------------------------------------
#------------------------------------------------------------------------------
n=40
m=2*n
N=n+2*m
#Initialization of the matrices and vectors
G=np.identity(n)
C=np.zeros((n,m))
d=np.zeros(m)
g=np.zeros(n)
for i in range(0,n):
    C[i,i]=1
    C[i,i+n]=-1
    g[i]=np.random.normal(0,1)
for i in range(0,m):
    d[i]=-10
#Let us solve the problem using np.linalg.solve
z0=np.zeros(N)
for i in range(n,N):
    z0[i]=1

res0,it0,condnumber0=Newton_mod(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it0)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it0)
for i in range(0,it0):
    iterations[i]=i
    printcondnumber[i]=condnumber0[i]
plt.figure(11)
plt.plot(iterations,printcondnumber,color='g')
plt.title("Cond.numb(np.linalg.solve), n=40")
plt.xlabel("Iterations")
plt.ylabel("Condition number")


res1,it1,condnumber1=Newton_mod_strat1(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it1)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it1)
for i in range(0,it1):
    iterations[i]=i
    printcondnumber[i]=condnumber1[i]
plt.figure(12)
plt.plot(iterations,printcondnumber,color='b')
plt.title("Cond.numb(ldl^t), n=40")
plt.xlabel("Iterations")
plt.ylabel("Condition number")    


res2,it2,condnumber2=Newton_mod_strat2(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it2)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it2)
for i in range(0,it2):
    iterations[i]=i
    printcondnumber[i]=condnumber2[i]
plt.figure(13)
plt.plot(iterations,printcondnumber,color='r')
plt.title("Cond.numb(cholesky), n=40")
plt.xlabel("Iterations")
plt.ylabel("Condition number") 
#------------------------------------------------------------------------------
#--------------------------------n=60------------------------------------------
#------------------------------------------------------------------------------
n=60
m=2*n
N=n+2*m
#Initialization of the matrices and vectors
G=np.identity(n)
C=np.zeros((n,m))
d=np.zeros(m)
g=np.zeros(n)
for i in range(0,n):
    C[i,i]=1
    C[i,i+n]=-1
    g[i]=np.random.normal(0,1)
for i in range(0,m):
    d[i]=-10
#Let us solve the problem using np.linalg.solve
z0=np.zeros(N)
for i in range(n,N):
    z0[i]=1

res0,it0,condnumber0=Newton_mod(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it0)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it0)
for i in range(0,it0):
    iterations[i]=i
    printcondnumber[i]=condnumber0[i]
plt.figure(14)
plt.plot(iterations,printcondnumber,color='g')
plt.title("Cond.numb(np.linalg.solve), n=60")
plt.xlabel("Iterations")
plt.ylabel("Condition number")


res1,it1,condnumber1=Newton_mod_strat1(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it1)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it1)
for i in range(0,it1):
    iterations[i]=i
    printcondnumber[i]=condnumber1[i]
plt.figure(15)
plt.plot(iterations,printcondnumber,color='b')
plt.title("Cond.numb(ldl^t), n=60")
plt.xlabel("Iterations")
plt.ylabel("Condition number")    


res2,it2,condnumber2=Newton_mod_strat2(G, C, d, g, n, z0,boolean=True)
iterations=np.zeros(it2)
iterations=iterations.astype(int)
printcondnumber=np.zeros(it2)
for i in range(0,it2):
    iterations[i]=i
    printcondnumber[i]=condnumber2[i]
plt.figure(16)
plt.plot(iterations,printcondnumber,color='r')
plt.title("Cond.numb(cholesky), n=60")
plt.xlabel("Iterations")
plt.ylabel("Condition number")   
    