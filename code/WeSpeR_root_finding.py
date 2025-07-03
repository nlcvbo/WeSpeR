# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, root_scalar

def line_search_p(f,a,b,eps=1e-1):
    max_iter = 1e3
    n_iter = 0
    ur = (a+b)/2
    ul = (a+b)/2
    while f(ul) < 0 and n_iter < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter += 1
    
    while f(ur) > 0 and n_iter < max_iter:
        ur  = b - (b-ur)/(1+eps)
        n_iter += 1
    
    return ur, ul, n_iter < max_iter

def line_search_m(f,a,b,eps=1e-1):
    max_iter = 1e3
    n_iter = 0
    ur = (a+b)/2
    ul = (a+b)/2
    while f(ul) > 0 and n_iter < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter += 1
    
    while f(ur) < 0 and n_iter < max_iter:
        ur  = b - (b-ur)/(1+eps)
        n_iter += 1
    
    return ur, ul, n_iter < max_iter

def line_search_r(f,a,b,eps=1e-1):
    max_iter = 1e3
    um = (a+b)/2
    n_iter0 = 0
    while not np.isfinite(f(um)):
        um  = a + (um-a)*(1+eps)
        n_iter0 += 1
        
    ur = um
    ul = um
    
    n_iter1 = 0
    while f(ul) > 0 and n_iter1 < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter1 += 1
    
    n_iter2 = 0
    while f(ur) < 0 and n_iter2 < max_iter:
        ur  = a + (ur- a)*(1+eps)
        n_iter2 += 1
    
    return ur, ul, (n_iter1 < max_iter) and (n_iter2 < max_iter)

def root_brentq_p(f,a,b,eps):
    ur, ul, success = line_search_p(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_m(f,a,b,eps):
    ur, ul, success = line_search_m(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_r(f,a,b,eps):
    ur, ul, success = line_search_r(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

