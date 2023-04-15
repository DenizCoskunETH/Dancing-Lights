# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 20:34:04 2022

@author: deniz
"""

import numpy as np 
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.optimize import fsolve
#from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
#import LissaJouData as Data
import image_process_lissa_jou as ip
import os
import glob

import matplotlib.cm as cm



def circle(t): 
    first_co = np.cos(t)
    sec_co = np.cos(t+np.pi/2)
    ret = np.array([first_co, sec_co])
    return ret


def Gamma(t, w, A, B, delta, l_1, l_2): #to_fit = [A, B, delta]
    first_co = np.cos(w*t)*B
    sec_co = np.cos(w*t+delta)*A
    ret = np.array([first_co+l_1, sec_co+l_2])
    return ret

def Gamma_2(t, w, A, B, delta): #to_fit = [A, B, delta]
    first_co = np.cos(w*t)*B
    sec_co = np.cos(w*t+delta)*A
    ret = np.array([first_co, sec_co])
    return ret

def distance(gamma_mes, gamma_to_fit, intervall, w, A, B, delta, l_1, l_2): 
    t_min = intervall[0]
    t_max = intervall[1]

    
    def min_dist(i):
        interval = np.linspace(t_min, t_max, 200)
             
        dist = lambda t: np.linalg.norm(gamma_mes[i]-gamma_to_fit(t, w, A, B, delta, l_1, l_2))**2
        dist_int = np.empty_like(interval)

        for k in range(len(interval)): 
            dist_int[k] = dist(interval[k])
        min_dist = min(dist_int)
        return min_dist
    
    sum_ = 0
    for i in range(gamma_mes[:,0].size):
        sum_ += min_dist(i)
    return sum_

def distance_2(gamma_mes, gamma_to_fit, intervall, w, A, B, delta): 
    t_min = intervall[0]
    t_max = intervall[1]

    
    def min_dist(i):
        interval = np.linspace(t_min, t_max, 200)
             
        dist = lambda t: np.linalg.norm(gamma_mes[i]-gamma_to_fit(t, w, A, B, delta))**2
        dist_int = np.empty_like(interval)

        for k in range(len(interval)): 
            dist_int[k] = dist(interval[k])
        min_dist = min(dist_int)
        return min_dist
    
    sum_ = 0
    for i in range(gamma_mes[:,0].size):
        sum_ += min_dist(i)
    return sum_

def Fit(f, gamma_mes, guess):
    omeg = str(f)
    w = 1
    plt.figure()
    plt.title("$f$ ="+omeg+"Hz")
    plt.plot(gamma_mes[:,0], gamma_mes[:,1], 'x', label = "meas", color = "red")
    T = (2*np.pi)/w
    
    def distance_for_fit_wiorigin(fit_param):
        A_ = fit_param[0]
        B_ = fit_param[1]
        delta_ = fit_param[2]
        l_1_ = fit_param[3]
        l_2_ = fit_param[4]
        res = distance(gamma_mes, Gamma, [0, T], w, A_, B_, delta_, l_1_, l_2_)
        return res
    
    def distance_for_fit_woorigin(fit_param):
        A_ = fit_param[0]
        B_ = fit_param[1]
        delta_ = fit_param[2]
        res = distance_2(gamma_mes, Gamma_2, [0, T], w, A_, B_, delta_)
        return res
    
    #FITTED = fsolve(distance_for_fit, guess)
    FITTED = minimize(distance_for_fit_wiorigin, guess)
    L_1 = FITTED.x[3]
    L_2 = FITTED.x[4]
    guess_2 = FITTED.x[0:4]
    FITTED = minimize(distance_for_fit_woorigin, guess_2)
    #plot from this point on 
    #print(FITTED)
    FITTED = FITTED.x
    A = FITTED[0]
    B = FITTED[1]
    delta = FITTED[2]

    time = np.linspace(0, (2*np.pi)/w, 200)
    
    y = Gamma(time, w, guess[0],guess[1],guess[2], guess[3], guess[4])
    x = Gamma(time, w, A, B, delta, L_1, L_2)
    x_1 = x[0,:]
    x_2 = x[1,:]
    y_1 = y[0,:]
    y_2 = y[1,:]
    print(L_1, L_2)
    plt.plot(y_1, y_2, '--', label = "guess", color = "black", alpha = 0.2)
    plt.plot(x_1, x_2, label = "fit", color = "black")
    plt.legend()
    plt.show()
    return A, B, delta
  











