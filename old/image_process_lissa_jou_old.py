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

def crop_image(image, origin, min_y, max_y, min_x, max_x):
#This function crops to the interesting part of the image
    min_x = int(min_x*len(image[:,0,0]))
    max_x = int(max_x*len(image[:,0,0]))
    min_y = int(min_y*len(image[0,:,0]))
    max_y = int(max_y*len(image[0,:,0]))
    new_image = image[min_x:max_x, min_y:max_y , :]
    start = np.array([min_x, min_y])
    
    new_origin = np.array([origin[1]-start[1], origin[0]-start[0]])
    
    return new_image, new_origin
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
    
def mark_boundaries(array, limit, minimum_number):
    success = True
    k = 0
    
    len_x = array[:,0].size
    len_y = array[0,:].size
    boundaries = np.zeros_like(array)
    for i in range(len_x): #!
        for j in range(len_y):
            if array[i,j] > limit:
                boundaries[i,j] = 1
                k+=1
    if k < minimum_number:
        success = False
    
    return boundaries, success
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

#!!!!!These two functions are meant to take cropped image and new origin
def get_path_gradient(image, origin, limit, minimum_number):
    print("Doing Gradient Method")
    image = get_color(image, 0) #taking red values
    abs_grad = abs_gradient(image)
    boundaries, success = mark_boundaries(abs_grad, limit, minimum_number)
    
    if success == False:
        print("grad method was unsec")
        return [0], False   #this means grad method did not work

#Take out after DEBUG
    fig = plt.figure()
    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(boundaries, interpolation='nearest', cmap=cm.Greys_r)
    plt.title("Chosen Pixels by Gradient Method")
    plt.show()
###

    path = create_path(boundaries)
    path -= origin
    
    return path, True 

def get_path_intensity(image, origin, width_origin, max_ratio, min_int):
    print("Doing Intensity Method")
    image = delete_origin(image,width_origin,width_origin,origin[0],origin[1])
    red_im = red_0_1(image, max_ratio, min_int)

#Take out after DEBUG
    fig = plt.figure()
    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(red_im, interpolation='nearest', cmap=cm.Greys_r)
    plt.title("Chosen Pixels by Intensity Method")
    plt.show()
###
   
    path = create_path(red_im)
    path -= origin
    
    return path
#!!!!

def get_path_method(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
             uncropped_origin, limit, minimum_number, width_origin, max_ratio, min_int,
             method):
#Gets the path with given method from uncropped image
   
    image, origin = crop_image(uncropped_image, uncropped_origin, crop_y_min, crop_y_max, crop_x_min, crop_x_max)

#This is to see the produced cropped image take out after debug 
    fig = plt.figure()
    ax2 = fig.add_subplot(122)
    ax2.imshow(image, interpolation='nearest', cmap=cm.Greys_r)
    plt.plot(origin[0], origin[1], 'x')
    plt.show()
##########
        
    if method == 'gradient':
        return get_path_gradient(image, origin, limit, minimum_number)
        
    if method == 'intensity':
        return get_path_intensity(image, origin, width_origin, max_ratio, min_int)
    
    else:
        return False
    
def get_path_combined(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
             uncropped_origin, limit, minimum_number, width_origin, max_ratio, min_int, reduce_index):
    
    path_grad, success = get_path_method(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, uncropped_origin, limit, minimum_number, width_origin, max_ratio, min_int
                           , 'gradient')
    if success == False:
        path_int = get_path_method(uncropped_image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, uncropped_origin, limit, minimum_number, width_origin, max_ratio, min_int
                               , 'intensity')
        return reduce(reduce_index, path_int)
    else:
        return path_grad

###############################################################################
###############################################################################


    


