# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:28:31 2023

@author: deniz
"""
import numpy as np 
import matplotlib.pyplot as plt
import image_process_lissa_jou as ip
import Lissa_Jou_Set_Fit02 as FIT
import Get_Spectrum as SPECTR 
import os
import glob
import time
import matplotlib.cm as cm
###########################INFO###############################################
#
###General Info:
#
#Directory of Images:
img_dir_0 = r'C:\Users\deniz\OneDrive\Masaüstü\Dancing Lights 2\trial_14_04_2023'
#img_dir_0 = r'C:\Users\deniz\OneDrive\Masaüstü\Dancing Lights 2\_measurement-14.04.2023\_measurement-14.04.2023'
#Frequency Array:
f_0 = np.array(range(180, 227))
#Cropping to the relevant part:
crop_x_min_0 = 0.26         #up down      
crop_x_max_0 = 0.86
crop_y_min_0 = 0.2          #left right
crop_y_max_0 = 0.95
#Reducing the number of Points in the choosen Path:
reduce_0 = 1            
#Position of the Origin
#origin_0 = [2384, 1611]    #position of origin
#origin_0 = [0, 0]    #position of origin
#
###intensity Method Fine Tuning:
#
width_origin_0 = 1000     #width of origin #became unimportant change in ip
#                 
max_ratio_0 = 0.7    #CONDITION: (bluegreen[i][j]/red[i][j])**3 < max_ratio and red[i][j] > min_int
min_int_0 = 170
#

#
###grad method fine tuning:
#
max_abs_grad_0 = 0.2#0.35   #[1 if array[i,j] > limit else 0 for i in range(len_x)] for j in range(len_y)]
min_number_0 = 0

#
##Data Save directory
#
data_text_0 = r'C:\Users\deniz\OneDrive\Masaüstü\Dancing Lights 2\Lissa Jou Fitting\data_trial_save.txt'
#######################INFO END################################################

omega, A_cal, B_cal, delta_cal = SPECTR.get_spectrum(img_dir_0, width_origin_0, crop_x_min_0, 
                                              crop_x_max_0, crop_y_min_0, crop_y_max_0, max_ratio_0, 
                                              min_int_0, f_0, reduce_0, max_abs_grad_0, min_number_0,
                                              data_text_0)




