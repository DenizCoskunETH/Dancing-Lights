# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:56:37 2023

@author: deniz

GIVE DIRECTORY FREQUENCY RANGE ORIGIN AND INTENSITIES
"""
import numpy as np 
import matplotlib.pyplot as plt
import image_process_lissa_jou as ip
import Lissa_Jou_Set_Fit02 as FIT 
import os
import glob
import time
import matplotlib.cm as cm

def Get_Fit(file, frequency, crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
              limit, minimum_number, width_origin, max_ratio, min_int, reduce_index):
#One Picture One single Freq returns Fit as well as time
    time_begin_fit = time.time()
    
    image = ip.read_bmp(file, 0, 1, 0, 1)
    path = ip.get_path_combined(image, crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
              limit, minimum_number, width_origin, max_ratio, min_int, reduce_index)
    
    Guess = np.array([path[0,0], path[0,1], 1.0, 0, 0])
    A, B, delta = FIT.Fit(frequency, path, Guess)
    
    time_end_fit = time.time()
    Delta_time = time_end_fit - time_begin_fit
    
    return A, B, delta, Delta_time


def get_spectrum(img_dir, width_origin, crop_x_min, crop_x_max, crop_y_min, crop_y_max, 
                 max_ratio, min_int, freq_range, reduce_index, limit_grad, minimum_number, 
                 text_to_save_dir):
    
    time_spec_begin = time.time()
    omega = freq_range*2*np.pi
    A = np.empty_like(omega)
    B = np.empty_like(omega)
    delta = np.empty_like(omega)


    data_path = os.path.join(img_dir,'*g') 
    files = glob.glob(data_path)

    for i in range(len(files)):
        print("+")
        A[i], B[i], delta[i], single_time = Get_Fit(files[i], freq_range[i], crop_y_min, crop_y_max, crop_x_min, crop_x_max, 
                                                     limit_grad, minimum_number, width_origin, max_ratio, min_int, reduce_index)
        print("Time forSingle Fit:", single_time)
        
        
        ########>>>>>>crash_safe_data_saving>>>>>>############
    
        f = open("data.txt", "a")
        f.write(i,'\t' )
        f.write(f[i],'\t' )
        f.write(A[i],'\t' )
        f.write(B[i],'\t' )
        f.write(delta[i],'\t' )
        f.write("\n")
        f.close()
    
        #########<<<<<<crash_safe_data_saving<<<<<###########
    
        
    plt.figure()
    plt.title("$A(\omega)$")
    plt.plot(omega, A)
    plt.show()
    plt.figure()
    plt.title("$B(\omega)$")
    plt.plot(omega, B)
    plt.show()
    plt.figure()
    plt.title("$\delta(\omega)$")
    plt.plot(omega, delta)
    plt.show()
    
    time_spec_end = time.time()
    

    print("omega=", omega)
    print("_____________________")
    print("A=", A)
    print("_____________________")
    print("B=", B)
    print("_____________________")
    print("delta=", delta)
    print("_____________________")
    print("the calibration took:")
    print(time_spec_end-time_spec_begin, "seconds")
    
    return omega, A, B, delta














