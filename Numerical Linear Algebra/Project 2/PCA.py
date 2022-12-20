#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:51:39 2022

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#--------------------------- Project 2 NLA-------------------------------------
#------------------------David Rosado Rodriguez--------------------------------
#------------------------------------------------------------------------------
#----------------------------------PCA-----------------------------------------


#Function that read data from exaple.dat
def read_txt():
    X = np.genfromtxt('example.dat', delimiter = ' ')
    return X.T

#Function thtat read data from the csv file
def read_csv():
    X = np.genfromtxt('RCsGoff.csv', delimiter = ',')
    #Get rid of unnecessary variables
    return X[1:,1:].T

#Function that reproduces the Scree plot   
def Scree_plot(S,number_figure,matrix_type):
    if matrix_type == 1:#covariance matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S**2)
        for i in range(len(S)):
            plt.scatter(i,S[i]**2,color='r')
        plt.title('Scree plot for the covariance matrix')
        plt.xlabel('Principal Components number')
        plt.ylabel('Eigenvalues')
        plt.show()
    else:#correlation matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S**2)
        for i in range(len(S)):
            plt.scatter(i,S[i]**2,color='r')
        plt.title('Scree plot for the correlation matrix')
        plt.xlabel('Principal Components number')
        plt.ylabel('Eigenvalues')
        plt.show()
#Function that computes the 3/4 of the total variance rule
def rule_34(var):
    partial_var = 0
    i=0
    while partial_var < 3/4:
        partial_var += var[i]
        i+=1
    return i
#Function that computes the Kasier rule
def Kasier(S):
    count = 0
    for i in range(len(S)):
        if S[i]**2>1:
            count += 1
    return count
    
#Function that apply PCA analysis
def PCA(matrix_choice, file_choice):
    
    #Choose the data
    if file_choice == 1:#text dataset
        X = read_txt()
    else:#csv dataset
        X = read_csv()
    
    #Substract the mean
    X = X - np.mean(X, axis = 0)
    n = X.shape[0]
    #Choose the matrix and complete the program
    if matrix_choice == 1:#covariance matrix
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        
        #Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)
        #Standard deviation of each of the PC
        #Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH, axis = 0)
        
        #Expression of the original dataset in the new PCA coordinates
        new_expr = np.matmul(VH,X).T
    else:#correlation matrix
        X = (X.T / np.std(X, axis = 1)).T
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        
        #Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)
        
        #Standard deviation of each of the PC
        #Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH.T, axis = 0)
        
        #Expression of the original dataset in the new PCA coordinates
        new_expr = np.matmul(VH,X).T
    return total_var, standard_dev, new_expr,S



if __name__ == '__main__':
    
    #-----------------------Analysis of the fist dataset-----------------------
   
    print('---------------Analysis of the example.dat dataset-----------------')
    print('\n')
    print('----------------------Covariance matrix----------------------------')
    total_var,standar_dev,new_expr,S = PCA(1,1)
    print('\n')
    print('Total variance accumulated in each of the principal components:\n',total_var)
    print('\n')
    print('Standard deviation of each of the principal components:\n',standar_dev)
    print('\n')
    print('Expression of the original dataset in the new PCA coordinates:\n',new_expr)
    Scree_plot(S,1,1)
    print('\n')
    print('Kasier rule:',Kasier(S))
    print('3/4 rule:',rule_34(total_var))
    print('\n')
    
    
    print('----------------------Correlation matrix----------------------------')
    total_var,standar_dev,new_expr,S = PCA(0,1)
    print('\n')
    print('Total variance accumulated in each of the principal components:\n',total_var)
    print('\n')
    print('Standard deviation of each of the principal components:\n',standar_dev)
    print('\n')
    print('Expression of the original dataset in the new PCA coordinates:\n',new_expr)
    Scree_plot(S,2,0)
    print('\n')
    print('Kasier rule:',Kasier(S))
    print('3/4 rule:',rule_34(total_var)) 
    print('\n')       
    
    #-----------------------Analysis of the second dataset---------------------
    
    print('---------------Analysis of the RCsGoff.csv dataset-----------------')
    print('\n')
    print('----------------------Covariance matrix----------------------------')
    total_var,standar_dev,new_expr,S = PCA(1,0)
    print(new_expr.shape)
    print('\n')
    print('Total variance accumulated in each of the principal components:\n',total_var)
    print('\n')
    print('Standard deviation of each of the principal components:\n',standar_dev)
    print('\n')
    print('Expression of the original dataset in the new PCA coordinates:\n',new_expr)
    Scree_plot(S,3,1)
    print('\n')
    print('Kasier rule:',Kasier(S))
    print('3/4 rule:',rule_34(total_var))
    print('\n')
    
    
    
    
    
    
    
    
    
    
    