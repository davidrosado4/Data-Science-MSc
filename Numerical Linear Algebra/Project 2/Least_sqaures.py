#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:01:41 2022

@author: david
"""

import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular,qr
#------------------------------------------------------------------------------
#--------------------------- Project 2 NLA-------------------------------------
#------------------------David Rosado Rodriguez--------------------------------
#------------------------------------------------------------------------------
#--------------------------Least Squares problem-------------------------------

#-------------------------Defintions of functions------------------------------

#Function that read the datafile from Virtual Campus, called 'dades', and creates
# the matrix A and the vector b for the Least Sqaures problem
def load_dades(degree):
    data = np.genfromtxt('dades', delimiter=' ')#read the data but with two columns of infinite values, we have to remove them
    points = np.zeros((data.shape[0],2))
    for i in range(0,data.shape[0]):
        points[i,0] = data[i,0]
        points[i,1] = data[i,3]
    #Let us construct the matrix A and the vector b 
    A = np.zeros((points.shape[0],degree))
    for i in range(0,points.shape[0]):
        A[i,:] = [points[i,0]**d for d in range(0,degree)]
    b = points[:,1]
    return A,b,points
#Function that read the datafile2.csv from Virtual Campus, called 'dades_regressio.csv'
#and construct the matrix A and the vector b for the Least Squares problem
def load_csv():
    X = np.genfromtxt('dades_regressio.csv', delimiter=',')
    A, b = X[:,:-1], X[:,-1]
    return A,b
#Function that solves the Least Sqaures problem with minimum norm using SVD
def LS_sol_svd(M,b):
    return np.matmul(M,b)
#Function that computes the pseudoinvers of a function using SVD factorization
def pseudo(M):
    #First, compute the SVD of the matrix
    U,S,VH = svd(M,full_matrices = False)
    #Let us discard small singular values and build the inverse of S
    for i in range(0,S.shape[0]):
        if S[i]>1e-15:
            S[i] = 1/S[i]
        else:
            S[i]=0
    #The pseudoinvers is given by VH.T * S^{-1} * U.T
    pseudoinvers = np.matmul(np.matmul(VH.T,np.diag(S)),U.T)
    return pseudoinvers
#Function that computes the solution of the Least Squares problem using QR 
def LS_qr_sol(M,b):
    #The solution depends on the rank of the matrix
    r = np.linalg.matrix_rank(M)
    if r == M.shape[1]: #full rank matrix
        Q1, R1 = np.linalg.qr(M) #thin QR factorization
        y = np.dot(Q1.T,b)
        x = solve_triangular(R1,y)
    else: #rank deficient matrix
        print('Rank deficient matrix!')
        Q, R, P = qr(M, pivoting = True) #complete QR factorization
        R1 = R[:r,:r]
        c = np.dot(Q.T,b)[:r]
        u = solve_triangular(R1,c)
        v = np.zeros((M.shape[1]-r))
        x = np.matmul(np.eye(M.shape[1])[:,P],np.concatenate((u,v)))
    return x
#----------------------------------Main----------------------------------------
if __name__ == '__main__':
    
    
    
    
    #--------------------------------------------------------------------------
    #-----------------------First dataset, datafile----------------------------
    #--------------------------------------------------------------------------
   
    
   
    
   #-------------------------------SVD approach-------------------------------
    #Let us find the best degree for fitting
    print('-------------------------------------------------------------------')
    print('-----------------------First dataset, datafile----------------------')
    print('--------------------------------------------------------------------')
    print('-------------------------SVD APPROACH------------------------------')
    print('\n')
    svd_errors = []
    for d in range(2,12):
        A, b, points = load_dades(d)
        A_plus = pseudo(A)
        sol_svd = LS_sol_svd(A_plus,b)
        svd_errors.append(np.linalg.norm(np.dot(A,sol_svd)-b))
        #We want to find the one who has minimum error
        best_degree = np.argmin(svd_errors) + 2
        #Let us find the worst one
        worst_degree = np.argmax(svd_errors) + 2
    
    print('For the first datafile we have found the following:')
    A, b, points = load_dades(best_degree)
    A_plus = pseudo(A)
    best_sol_svd = LS_sol_svd(A_plus,b)
    print('Best degree:', best_degree)
    print('Solution norm:', np.linalg.norm(best_sol_svd))
    print('Error:', np.linalg.norm(np.dot(A,best_sol_svd)-b))
    print('\n') 
    #Of course, the more degree, the better we fit.(see the final result to understand this comment) 
    #If we print the errors, we can see that degree 6 is more than enough
    #print(svd_errors)
    
    #Let us plot the points and the polynomial for the best and the worst degree
    #and also for degree 6
    A, b, points = load_dades(worst_degree)
    A_plus = pseudo(A)
    worst_sol_svd = LS_sol_svd(A_plus,b)
    print('Worst degree:', worst_degree)
    print('Solution norm:', np.linalg.norm(worst_sol_svd))
    print('Error:', np.linalg.norm(np.dot(A,worst_sol_svd)-b))
    print('\n')
    A, b, points = load_dades(6)
    A_plus = pseudo(A)
    sol_6_svd = LS_sol_svd(A_plus,b)
    print('Solution norm for degree 6:' , np.linalg.norm(sol_6_svd))
    print('Error:', np.linalg.norm(np.dot(A,sol_6_svd)-b))
    print('\n')
    
    plt.figure(1)
    plt.scatter(points[:,0],points[:,1],s=5,c='b')
    x = np.linspace(0.8,8.2,100)
    y_best = 0
    for i in range(0,best_degree):
        y_best = y_best + best_sol_svd[i]*x**i
    y_worst = 0
    for i in range(0,worst_degree):
        y_worst = y_worst + worst_sol_svd[i]*x**i
    y_6 = 0
    for i in range(0,6):
        y_6 = y_6 + sol_6_svd[i]*x**i
    plt.plot(x,y_best,color='red', label ='best_degree (11)')
    plt.plot(x,y_worst,color='green', label = 'worst_degree (2)')
    plt.plot(x,y_6,color='purple', label = 'degree 6')
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('Least Squares Porblem with SVD')
    plt.show()
    
    #----------------------------------QR approach-----------------------------
    print('------------------------------QR APPROACH--------------------------')
    print('\n')
    #Let us find the best degree for fitting
    qr_errors = []
    for d in range(2,12):
        A, b, points = load_dades(d)
        sol_qr = LS_qr_sol(A,b)
        qr_errors.append(np.linalg.norm(np.dot(A,sol_qr)-b))
        #We want to find the one who has minimum error
        best_degree = np.argmin(qr_errors) + 2
        #Let us find the worst one
        worst_degree = np.argmax(qr_errors) + 2
    
    print('For the first datafile we have found the following:')
    A, b, points = load_dades(best_degree)
    best_sol_qr = LS_qr_sol(A,b)
    print('Best degree:', best_degree)
    print('Solution norm:', np.linalg.norm(best_sol_qr))
    print('Error:', np.linalg.norm(np.dot(A,best_sol_qr)-b))
    print('\n') 
    #Of course, the more degree, the better we fit.(see the final result to understand this comment) 
    #If we print the errors, we can see that degree 6 is more than enough
    #print(svd_errors)
    
    #Let us plot the points and the polynomial for the best and the worst degree
    #and also for degree 6
    A, b, points = load_dades(worst_degree)
    worst_sol_qr = LS_qr_sol(A,b)
    print('Worst degree:', worst_degree)
    print('Solution norm:', np.linalg.norm(worst_sol_qr))
    print('Error:', np.linalg.norm(np.dot(A,worst_sol_qr)-b))
    print('\n')
    A, b, points = load_dades(6)
    sol_6_qr = LS_qr_sol(A,b)
    print('Solution norm for degree 6:' , np.linalg.norm(sol_6_qr))
    print('Error:', np.linalg.norm(np.dot(A,sol_6_qr)-b))
    print('\n')
    
    plt.figure(2)
    plt.scatter(points[:,0],points[:,1],s=5,c='b')
    x = np.linspace(0.8,8.2,100)
    y_best = 0
    for i in range(0,best_degree):
        y_best = y_best + best_sol_qr[i]*x**i
    y_worst = 0
    for i in range(0,worst_degree):
        y_worst = y_worst + worst_sol_qr[i]*x**i
    y_6 = 0
    for i in range(0,6):
        y_6 = y_6 + sol_6_qr[i]*x**i
    plt.plot(x,y_best,color='red', label ='best_degree (11)')
    plt.plot(x,y_worst,color='green', label = 'worst_degree (2)')
    plt.plot(x,y_6,color='purple', label = 'degree 6')
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('Least Squares Porblem with QR')
    plt.show()


    #--------------------------------------------------------------------------
    #--------------------Second dataset, datafile2.csv-------------------------
    #--------------------------------------------------------------------------
   
    
    print('-------------------------------------------------------------------')
    print('------------------Second dataset, datafile2.csv---------------------')
    print('--------------------------------------------------------------------') 
    #-------------------------------SVD approach-------------------------------
    print('-------------------------SVD APPROACH------------------------------')
    print('\n')
    print('For the second datafile we have found the following:')
    A,b = load_csv()
    A_plus = pseudo(A)
    sol_svd = LS_sol_svd(A_plus,b)
    print('Solution norm :', np.linalg.norm(sol_svd))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd)-b))
    print('\n')
    #----------------------------------QR approach-----------------------------
    print('------------------------------QR APPROACH--------------------------')
    print('\n')
    print('For the second datafile we have found the following:')
    sol_qr = LS_qr_sol(A, b)
    print('Solution norm :', np.linalg.norm(sol_qr))
    print('Error:', np.linalg.norm(np.dot(A,sol_qr)-b))
    print('\n')
    



