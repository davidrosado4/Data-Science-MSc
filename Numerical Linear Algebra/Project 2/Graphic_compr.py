#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:18:16 2022

@author: david
"""

import numpy as np
from imageio.v2 import imread,imsave
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#--------------------------- Project 2 NLA-------------------------------------
#------------------------David Rosado Rodriguez--------------------------------
#------------------------------------------------------------------------------
#--------------------------Graphic compression---------------------------------

#Function that compresses an image
def compress(image,r,save):
    #Let us read the image and create a vector for the compressed one
    img = imread(image)
    compressed_img = np.zeros(img.shape)
    #we also will compute the relative error
    error=0
    #Since we will deal with color image, we have to work with three different matrices(RGB)
    for i in range(0,3):
        img_palette = img[:,:,i]
        #SVD factorization of the matrix
        U,S,VH = np.linalg.svd(img_palette)
        #Compressions, with the rate given
        compress = np.matmul(np.matmul(U[:,:r],np.diag(S[:r])),VH[:r,:])
        #Compute the relative error 
        error += np.linalg.norm(img_palette - compress) / np.linalg.norm(img_palette)
        compressed_img[:,:,i] = compress
    #Correct the ranges
    compressed_img = np.rint(255*(compressed_img - np.min(compressed_img))/(np.max(compressed_img) - np.min(compressed_img)))
    #Save some of the images
    if save == True:
        imsave('compressed' + "-" + str(r) + "_" +str((error*100)/3.0) + "_" + ".jpg", compressed_img.astype(np.uint8))
    return (error*100)/3

if __name__ == '__main__':
    
    #--------------------------Landscape---------------------------------------
    #Let us compress images, using different compressing rate
    graduacio_errors = []
    for r in range(10,601,10):
        boolean = False
        if r % 100 == 0 or r == 10 or r == 50 or r == 150 or r == 250 or r == 250 or r == 350 or r == 450 or r == 550:
            boolean = True
        error = compress('landscape.jpg',r,boolean)
        graduacio_errors.append(error)
    plt.plot(range(10,601,10), graduacio_errors)  
    plt.title('Image error for the landscape')
    plt.xlabel('rate')
    plt.ylabel('error(%)')
    plt.show()
    
    #--------------------------Graduation---------------------------------------
    #Let us compress for four specific values, due to the big image size
    #otherwise it would take a long computational time
    print('-----------------Picture of my graduation-------------------------')
    print('------------------------------------------------------------------')
    error = compress('graduacio.jpg',10,True)
    print('The compression error for rate 10 is:', error,'%')
    for r in range(100,501,200):
        error = compress('graduacio.jpg', r, True)
        print('The compression error for rate', r,'is:', error,'%')
        
    #--------------------------Picture of you!---------------------------------
    #Let us compress for four specific values, due to the big image size
    #otherwise it would take a long computational time
    print('-----------------Picture of you(Arturo)---------------------------')
    print('------------------------------------------------------------------')
    error = compress('vieiro.jpg',10,True)
    print('The compression error for rate 10 is:', error,'%')
    for r in range(100,501,200):
        error = compress('vieiro.jpg', r, True)
        print('The compression error for rate', r,'is:', error,'%')    
    
    














    