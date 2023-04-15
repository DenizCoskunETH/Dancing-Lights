# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:44:16 2022

@author: moritz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp
import scipy.ndimage
import os
import glob

############General Functions for Getting Image###############################
##############################################################################
def read_bmp(path_to_file, min_x, max_x, min_y, max_y):
#function for reading bitmap file
    image = plt.imread(path_to_file)
    min_x = int(min_x*len(image[:,0,0]))
    max_x = int(max_x*len(image[:,0,0]))
    min_y = int(min_y*len(image[0,:,0]))
    max_y = int(max_y*len(image[0,:,0]))
    image = image[min_x:max_x, min_y:max_y, :]
    image = image.copy()
    return image

def get_color(image, color_index):
#function for extracting the value of a pixel x and y for a given color
#0-red, 1-green, 2-blue
    return image[:,:,color_index]

def get_pixel(image, x, y, color):
    return image[x][y][color]

def crop_image(image, min_y, max_y, min_x, max_x):
#This function crops to the interesting part of the image
    min_x = int(min_x*len(image[:,0,0]))
    max_x = int(max_x*len(image[:,0,0]))
    min_y = int(min_y*len(image[0,:,0]))
    max_y = int(max_y*len(image[0,:,0]))
    new_image = image[min_x:max_x, min_y:max_y , :]
    #start = np.array([min_x, min_y])
    
    #new_origin = np.array([origin[1]-start[1], origin[0]-start[0]])
    
    return new_image
############################################################
############################################################

###########Intensity Method Functions#########################################
##############################################################################

def delete_origin(image, dx, dy, x, y ):
    # image: original uncropped image
    # dx, dy width of the block
    # x, y coordinates of origin
    # return image with origin blacked out
    
    
    darkness = [0,0,0]
    
    for i in range(x-dx,x+dx):
        for j in range(y-dy, y+dy):
            if (abs(x-i)**2 + abs(y-j)**2) < dx**2:
                image[i,j] = darkness
    return image

def mean(array):
    return sum(array)/len(array)

def delete_origin_01(matrix, blocksize_radius):
    #input give a black (0) and white(1) matrix that is the already processed image
    #blocksize_radius defines the area around the origin that is replaced with black pixels
    
    #returntype: ---> the function edits the input matrix directly
    #            ---> however it returns the cooordinates of the origin aka center of mass
    
    
    
    len_x = matrix[:,0].size
    len_y = matrix[0,:].size
    # calculate the center of mass so to speak:
        #get all the coordinates from white pixels
    x_coordinates = np.array([i for i in range(len_x) for j in range(len_y) if matrix[i,j] == 1])
    y_coordinates = np.array([j for i in range(len_x) for j in range(len_y) if matrix[i,j] == 1])

    #use abbreviation cms = center of mass
    cms_x = int(mean(x_coordinates))
    cms_y = int(mean(y_coordinates))
    
    #correspondence of origin and center of mass
    origin = np.array([cms_x,cms_y])
    
    #now delete everything within this radius:
        
    for i in range(-blocksize_radius,blocksize_radius):
        for j in range(-blocksize_radius, blocksize_radius):
            if i**2 + j**2 <= blocksize_radius:
                matrix[cms_x+i,cms_y+j] = 0
    
    return matrix, origin
                
                
    
def smoothen(matrix, sigma_ = 7.0):
    sigma = [sigma_, sigma_]
    y = sp.ndimage.filters.gaussian_filter(matrix, sigma, mode='constant')
    return y


def red_0_1(image, max_ratio, min_int):
    red = get_color(image, 0)
    red = smoothen(red)
    green = get_color(image, 1)
    green = smoothen(green)
    blue =get_color(image, 2)
    blue = smoothen(blue)
    bluegreen = (blue + green)/2
    new = np.zeros_like(red)
    
                
    new = np.array([[1 if (bluegreen[i][j]/red[i][j])**3 < max_ratio and red[i][j] > min_int else 0 for j  in range(len(red[0,:]))] for i in range(len(red[:,0])) ]) 

    return new

#####################################################################
#####################################################################

#####Gradient Method Functions#################################################
###############################################################################    
def gradient(array): 
    return np.gradient(array)

def abs_gradient(array):
    grad_image = np.gradient(array)
    grad_image_abs = grad_image[0]**2+grad_image[1]**2
    maximum = np.max(grad_image_abs)
    return grad_image_abs/maximum
    
def mark_boundaries(array, limit):
    #k = 50
    
    len_x = array[:,0].size
    len_y = array[0,:].size
    #the following loop was replaced by an equivalent on liner below:
    # boundaries = np.zeros_like(array)
    # for i in range(len_x): #!
    #     for j in range(len_y):
    #         if array[i,j] > limit:
    #             boundaries[i,j] = 1
    #             k+=1
    
    boundaries = np.array([ [1 if array[i,j] > limit else 0 for i in range(len_x)] for j in range(len_y)])
    boundaries = boundaries.T    
    #k = np.count_nonzero(boundaries == 1)
    #[entry if tag in entry else [] for tag in tags for entry in entries]
    #print(k)
    #if k < minimum_number:
        #success = False
    
    return boundaries
###############################################
###############################################

######Getting Path Assuming we have the Right Binary Matrix###################
##############################################################################

def create_path(binary_matrix, const = 1): #pixel to real distance ratio
#takes binary matrix creates the real path where ones are
    array = const*np.array([np.array([len(binary_matrix)-i,j]) for i in range(len(binary_matrix)) for j in range(len(binary_matrix[0,:])) if binary_matrix[i][j]==1])
    
    return array

def reduce(n,data):
    new = np.array([data[i] for i in range (len(data[:,0])) if i%n == 0])
    return new

##############################################################################
##############################################################################

#####Getting Path with Both Methods###########################################
##############################################################################
def get_0_1_combined(image, limit, max_ratio, min_int):
    grad_image = get_color(image, 0)+get_color(image,1)+get_color(image,2)
    grad_image = abs_gradient(grad_image)
    grad_0_1 = mark_boundaries(grad_image, limit)
    int_0_1 = red_0_1(image, max_ratio, min_int)
    
    image_0_1 = (1/2)*(grad_0_1+int_0_1)
    image_0_1 = np.array(image_0_1, dtype = int)
    
    
    return image_0_1
    
    


    
def get_path_combined(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
             limit, minimum_number, width_origin, max_ratio, min_int, reduce_index):
    
    image = crop_image(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max)
    
    fig = plt.figure()
    ax2 = fig.add_subplot(122)
    ax2.imshow(image, interpolation='nearest', cmap=cm.Greys_r)
    #plt.plot(origin[0], origin[1], 'x')
    plt.show()
    
    image_0_1 = get_0_1_combined(image, limit, max_ratio, min_int)
    
    fig = plt.figure(dpi = 400)
    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(image_0_1, interpolation='nearest', cmap=cm.Greys_r)
    #plt.plot(origin[1],origin[0],'x')
    plt.title("Chosen Pixels by both methods")
    plt.show()
    
    
    image_0_1, origin = delete_origin_01(image_0_1, width_origin)
    
    fig = plt.figure(dpi = 400)
    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(image_0_1, interpolation='nearest', cmap=cm.Greys_r)
    plt.plot(origin[1],origin[0],'x')
    plt.title("Chosen Pixels by both methods")
    plt.show()
    
    path = create_path(image_0_1)
    path -= origin
    
    return path#reduce(reduce_index, path)
    

###############################################################################
###############################################################################


    


