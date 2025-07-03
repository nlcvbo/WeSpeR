# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from WeSpeR_root_finding import root_brentq_m, root_brentq_p, root_brentq_r

def mld_unif(x, alpha):
    return 1 + x/alpha*np.log(1 + alpha/(1-alpha/2-x))

def mld_unifp(x,alpha):
    return np.log(1 + alpha/(1-alpha/2-x))/alpha + x/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x))

def mld_unifpp(x,alpha):
    return 2/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x)) - 2*x*(x-1)/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x))**2

def x_unif(w,t,alpha,c):
    def xk(u):
        g1 = c*(w*t/(t-u)).sum()
        alpha_d = alpha/2
        m_init = 1 - 1/2/g1 -np.sqrt(1+4*g1*(g1-1)*alpha_d**2)/2/g1 # mldm for 2 diracs with alpha_d = alpha/2
        f = lambda x: mld_unif(x, alpha)-g1
        fp = lambda x: mld_unifp(x, alpha)
        m = root_scalar(f, x0=m_init, fprime=fp, method="newton").root
        
        rst = -u*g1*m
        return rst
    return xk

def x_unif_p(w,t,alpha,c):
    def xkp(u):
        g1 = c*(w*t/(t-u)).sum()
        alpha_d = alpha/2
        m_init = 1 - 1/2/g1 -np.sqrt(1+4*g1*(g1-1)*alpha_d**2)/2/g1 # mldm for 2 diracs with alpha_d = alpha/2
        f = lambda x: mld_unif(x, alpha)-g1
        fp = lambda x: mld_unifp(x, alpha)
        m = root_scalar(f, x0=m_init, fprime=fp, method="newton").root
        
        tw = c*(w*t/(t - u)).sum()
        twp = c*(w*t/(t - u)**2).sum()
        v = m
        vp = twp/mld_unifp(m,alpha)
        
        rst = -tw*v - u*twp*v - u*tw*vp
        return rst
    return xkp

def x_unif_pp(w,t,alpha,c):
    def xkpp(u):
        g1 = c*(w*t/(t-u)).sum()
        alpha_d = alpha/2
        m_init = 1 - 1/2/g1 -np.sqrt(1+4*g1*(g1-1)*alpha_d**2)/2/g1 # mldm for 2 diracs with alpha_d = alpha/2
        f = lambda x: mld_unif(x, alpha)-g1
        fp = lambda x: mld_unifp(x, alpha)
        m = root_scalar(f, x0=m_init, fprime=fp, method="newton").root
        
        tw = c*(w*t/(t - u)).sum()
        twp = c*(w*t/(t - u)**2).sum()
        twpp = 2*c*(w*t/(t - u)**3).sum()
        v = m
        vp = twp/mld_unifp(m,alpha)
        vpp = twpp/mld_unifp(m,alpha) - twp**2*mld_unifpp(m,alpha)/mld_unifp(m,alpha)**3
        
        rst = - twp*v - tw*vp - twp*v - u*twpp*v - u*twp*vp - tw*vp - u*twp*vp - u*tw*vpp
        return rst
    return xkpp

def tw(u, w, t, d, wd, c):
    return c*(w*t/(t - u)).sum()

def twp(u, w, t, d, wd, c):
    return c*(w*t/(t - u)**2).sum()

def twpp(u, w, t, d, wd, c):
    return 2*c*(w*t/(t - u)**3).sum()

def twppp(u, w, t, d, wd, c):
    return 6*c*(w*t/(t - u)**4).sum()

def v(u, k, w, t, d, wd, c, m):
    return m

def vp(u, k, w, t, d, wd, c, m):
    return twp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum()

def vpp(u, k, w, t, d, wd, c, m):
    return twpp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum() - twp(u, w, t, d, wd, c)**2*2*(wd*d/(d - m)**3).sum()/(wd*d/(d - m)**2).sum()**3

def vppp(u, k, w, t, d, wd, c, m): #see p.135
    return twppp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum() \
           - 3*twp(u, w, t, d, wd, c)*twpp(u, w, t, d, wd, c)*2*(wd*d/(d - m)**3).sum()/(wd*d/(d - m)**2).sum()**3 \
           + 3*twp(u, w, t, d, wd, c)**3*2*(wd*d/(d - m)**3).sum()**2/(wd*d/(d - m)**2).sum()**5 \
           - twp(u, w, t, d, wd, c)**3*6*(wd*d/(d - m)**4).sum()/(wd*d/(d - m)**2).sum()**4

def find_support_unif(c, alpha, t, w, eps=1e-1, verbose = False):
    # eps >= 1e-1 necessary
    T = t.shape[0]
    support = []
    
    for i in range(T-1):
        xkpp = x_unif_pp(w,t,alpha,c)
        xkp = x_unif_p(w,t,alpha,c)
        xk = x_unif(w,t,alpha,c)
            
        try:
            u0 = root_brentq_p(xkpp,t[i],t[i+1],eps)
            if xkp(u0) == 0:
                support += [xk(u0),xk(u0)]
                if verbose:
                    print(u0,xk(u0),xkp(u0))
            elif xkp(u0) > 0:
                try:
                    u1 = root_brentq_m(xkp,t[i],u0,eps)
                    u2 = root_brentq_p(xkp,u0,t[i+1],eps)
                    support += [xk(u1),xk(u2)]
                    if verbose:
                        print(u1,xk(u1),xkp(u1))
                        print(u2,xk(u2),xkp(u2))
                except ValueError:
                    if verbose:
                        print("Error",t[i],t[i+1])
        except ValueError:
            if verbose:
                print("Error:",t[i],t[i+1])
        # otherwise, no spectral gap
        
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M
    xkp = x_unif_p(w,t,alpha,c)
    xk = x_unif(w,t,alpha,c)
    try:
        u0 = root_brentq_p(xkp,0,t[0],eps)
        support += [0,xk(u0)]
        if verbose:
            print(u0,xk(u0),xkp(u0))
    except ValueError:
        if verbose:
            print("Error",0,t[0])
    
    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_unif_pp(w,t,alpha,c)
    xkp = x_unif_p(w,t,alpha,c)
    xk = x_unif(w,t,alpha,c)
    
    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1]+1, eps)
        support += [xk(u0)]
        if verbose:
            print(u0,xk(u0),xkp(u0))
    except ValueError:
        if verbose:
            print("Error:",t[-1],np.inf)
    
    support.sort()
    return support[1:]