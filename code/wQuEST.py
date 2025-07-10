import torch
import scipy as sp
import numpy as np
from WeSpeR_support_identification_unif import find_support_unif_u, mld_unif, mld_unifp
from WeSpeR_support_identification_ewma import find_support_exp_u, mld_exp, mld_expp
from WeSpeR_support_identification_Ndiracs import find_support_Ndiracs_u, mld_Ndiracs, mld_Ndiracsp
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import time

def m_tilde_loss_w(ab, x, w, ww, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    u = 1 + c*w[None,:]*(wt[None,:]*(t[None,:]*a[:,None]-x[:,None])*t[None,:]/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-torch.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    v = - c*w[None,:]*(wt[None,:]*t[None,:]**2*(-torch.exp(b)[:,None])/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-torch.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    loss = (a - (w[None,:]*ww[None,:]*u/(u**2+v**2)).sum(axis=1))**2 + (-torch.exp(b) + (w[None,:]*ww[None,:]*v/(u**2+v**2)).sum(axis=1))**2
    return loss.mean().to(float)

def X_loss_w(ab, x, w, ww, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    loss = torch.abs(X + (ww[None,:]*w[None,:]/(x[:,None] - c*w[None,:]*(wt[None,:]*t[None,:]/(1 + t[None,:]*X[:,None])).sum(axis=1)[:,None])).sum(axis=1))**2
    return loss.mean().to(float)

def m_X_w(x, w, ww, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = torch.tensor([0,0])
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w, x0 = x0, args = (x[i:i+1], w, ww, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = ab
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = torch.tensor([0,0])
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-5*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w, x0 = torch.tensor([0,0]), args = (z, w, ww, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w, x0 = x0, args = (x[i:i+1], w, ww, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = torch.tensor([0,0])
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def m_tilde_loss_w_ewma(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    m_tilde = a - torch.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*m_tilde[:,None]-x[:,None])).sum(axis=1)
    integral = mld_exp(-1/theta, alpha)/theta 
    loss = torch.abs(m_tilde - integral)**2
    return loss.mean().to(float)

def X_loss_w_ewma(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*X[:,None] + 1)).sum(axis=1)
    integral = mld_exp(x/theta, alpha)/theta 
    loss = torch.abs(X - integral)**2
    return loss.mean().to(float)

def m_X_w_ewma(x, alpha, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = torch.tensor([0,0])
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w_ewma, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = ab
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = torch.tensor([0,0])
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-3*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w_ewma, x0 = torch.tensor([(wt*t).sum(),0]), args = (z, alpha, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w_ewma, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = torch.tensor([0,0])
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def m_tilde_loss_w_unif(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    m_tilde = a - torch.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*m_tilde[:,None]-x[:,None])).sum(axis=1)
    integral = mld_unif(-1/theta, alpha)/theta 
    loss = torch.abs(m_tilde - integral)**2
    return loss.mean().to(float)

def X_loss_w_unif(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*X[:,None] + 1)).sum(axis=1)
    integral = mld_unif(x/theta, alpha)/theta 
    loss = torch.abs(X - integral)**2
    return loss.mean().to(float)

def m_X_w_unif(x, alpha, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = torch.tensor([0,0])
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w_unif, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = ab
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = torch.tensor([0,0])
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-5*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w_ewma, x0 = torch.tensor([0,0]), args = (z, alpha, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w_ewma, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = torch.tensor([0,0])
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def WeSpeR_support_u(tau, wtau, d, wd, c, weights, w_args, method = None, verbose = True):
    t = np.unique(tau)
    wt = np.zeros(t.shape)
    for i in range(len(t)):
        wt[i] = wtau[tau == t[i]].sum()
    
    if weights == "ewma":
        alpha = w_args[0]
        Kt = w_args[1]
        support, u, mldm, tildeomegai  = find_support_exp_u(c, alpha, t, wt, Kt = Kt, eps=1e-1, method=method, verbose = verbose)
        
    elif weights == "unif":
        alpha = w_args[0]
        Kt = w_args[1]
        support, u, mldm, tildeomegai  = find_support_unif_u(c, alpha, t, wt, Kt = Kt, eps=1e-1, method=method, verbose = verbose)
        
    elif weights == "Ndiracs":
        Kd = w_args[0]
        Kt = w_args[1]
        try:
            assert wd.shape == d.shape
        except:
            wd = np.ones(d.shape[0])/d.shape[0]
        support, u, mldm, tildeomegai = find_support_Ndiracs_u(c, wd, d, t, wt, Kt = Kt, Kd = Kd, eps=1e-2, method=method, verbose = verbose)
    elif weights == "test":
        support, u, mldm, tildeomegai = np.array([0.1, 10]), \
                                    np.array([-1/tau[0]-1,-1/tau[-1]+1]), \
                                    np.array([1-1/(-1/tau[0]-1),1-1/(-1/tau[-1]+1)]), \
                                    np.ones(1)
                                    
    return torch.from_numpy(support), torch.from_numpy(u), torch.from_numpy(mldm), torch.from_numpy(tildeomegai)

def WeSpeR_density(omega, support, d, wd, c, tau, tildeomegai, mu=0.1, wt = None, weights = 'test', w_args = None, verbose = True):
    if support.shape[0] % 2 != 0:
        support = np.concatenate([support, [tau[-1]*d[-1]*(1+np.sqrt(c))**2]])
        
    nu = support.shape[0]//2
    
    mu = max(0,min(mu,1))
    
    # Number of points
    omegai = (mu/nu + (1-mu)*tildeomegai)*omega
    
    # Discretization
    if omegai.sum() > 0:
        omegai = torch.round(omegai*omega/omegai.sum()+0.5).to(int)
    else:
        omegai = torch.round(omega*torch.ones(nu)/nu+0.5).to(int)
    
    try:
        assert wt.shape == tau.shape
    except:
        wt = torch.ones(tau.shape[0])/tau.shape[0]
    
    # Grid
    grid = torch.zeros(0)
    density = torch.zeros(0)
    X = torch.zeros(0)
    i_idx = torch.zeros(0, dtype=int)
    j_idx = torch.zeros(0, dtype=int)
    sin2 = torch.zeros(0)
    for i in range(nu):
        ul = support[2*i]
        ur = support[2*i+1]
        j = torch.ones(omegai[i]).cumsum(axis=0)
        xij = ul + (ur-ul)*torch.sin(np.pi*j/2/(omegai[i]+1))**2
        
        if weights == "ewma":
            alpha = w_args[0]
            m, Xij = m_X_w_ewma(xij, alpha, c, tau, wt)
            
        elif weights == "unif":
            alpha = w_args[0]
            m, Xij = m_X_w_unif(xij, alpha, c, tau, wt)
            
        elif weights == "Ndiracs":
            m, Xij = m_X_w(xij, d, wd, c, tau, wt)
            
        else:
            m, Xij = m_X_w(xij, d, wd, c, tau, wt)
            
        f = torch.abs(m.imag)/np.pi
        
        grid = torch.concatenate([grid, ul*torch.ones(1), xij, ur*torch.ones(1)])
        density = torch.concatenate([density, torch.zeros(1), f, torch.zeros(1)])
        X = torch.concatenate([X, torch.zeros(1), Xij, torch.zeros(1)])
        j_idx = torch.concatenate([j_idx, torch.ones(omegai[i]+2, dtype=int).cumsum(axis=0) - 1])
        i_idx = torch.concatenate([i_idx, 2*i*torch.ones(omegai[i]+2, dtype=int)])
        sin2 = torch.concatenate([sin2,  torch.zeros(1), torch.sin(np.pi*j/2/(omegai[i]+1))**2, torch.ones(1)])
    
    xi = grid.to(tau.dtype)
    f = density.to(tau.dtype)
    X = X
    
    F0 = max(0,1-1/c)
    G = torch.zeros(xi.shape, dtype=tau.dtype)
    F = torch.zeros(xi.shape, dtype=tau.dtype)
    G[0] = F0
    F[0] = F0
    for i in range(nu):
        j = torch.ones(omegai[i]+1, dtype=int).cumsum(0)
        offset = omegai[:i].sum(axis=0) + 2*i
        if offset > 0:
            G[offset] = F[offset-1]
            F[offset] = F[offset-1]
        G[j+offset] = G[offset] + 1/2*((xi[j+offset] - xi[j-1+offset])*(f[j+offset] + f[j-1+offset])).cumsum(axis=0)
        if tildeomegai.sum(axis=0) > 0:
            if (G[offset+j][-1] - F[offset]).to(float) > 0:
                F[j+offset] = F[offset] + tildeomegai[i]*(1-F[0])*(G[offset+j] - F[offset])/(G[offset+j][-1] - F[offset])
            else:
                F[j+offset] = F[offset] + tildeomegai[i]*(1-F[0])*j/j[-1]
        else:
            F[j+offset] = G[j+offset]
    if tildeomegai.sum(axis=0) == 0:
        F = G/G[-1]
    return xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu

class f_xi(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, method = 'poly', verbose = True):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        tau = input
        
        if d == None:
            d = torch.ones(1)
        if wd == None or wd.shape != d.shape:
            wd = torch.ones(d.shape[0])/d.shape[0]
        if wt == None:
            wt = torch.ones(tau.shape[0])/tau.shape[0]
        
        
        
        support, u, mldm, tildeomegai = WeSpeR_support_u(tau.detach().numpy(), wt.numpy(), d.numpy(), wd.numpy(), c, weights, w_args, method = method, verbose = verbose)
                
        # print(support)
        # print(torch.diff(support))
        
        xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu = WeSpeR_density(omega, support, d, wd, c, tau, tildeomegai, mu=mu, wt = wt, weights = weights, w_args = w_args, verbose = verbose)
        
        
        # For debug (test 4)
        # output = torch.concatenate([f, F, xi, X, support])
        
        output = torch.concatenate([f, F, xi, omegai, nu*torch.ones(1)])
        ctx.save_for_backward(input, output)
        ctx.wt = wt
        ctx.d = d
        ctx.wd = wd
        ctx.c = c
        ctx.weights = weights
        ctx.w_args = w_args
        ctx.omega = omega
        ctx.u = u
        ctx.mldm = mldm
        ctx.tildeomegai = tildeomegai
        ctx.omegai = omegai
        ctx.i_idx = i_idx
        ctx.j_idx = j_idx
        ctx.sin2 = sin2
        ctx.X = X
        ctx.xi = xi
        ctx.f = f
        ctx.G = G
        ctx.F = F
        ctx.nu = nu
        
        # if (1-torch.isfinite(input).to(int)).sum() != 0:
        #     print("NaN input")
        #     print(input)
            
            
        # if (1-torch.isfinite(output).to(int)).sum() != 0:
        #     print("NaN output")
        #     print(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, output = ctx.saved_tensors
        tau = input
        wt = ctx.wt
        d = ctx.d
        wd = ctx.wd
        c = ctx.c
        weights = ctx.weights
        w_args = ctx.w_args
        omega = ctx.omega
        u = ctx.u
        mldm = ctx.mldm
        omegai = ctx.omegai
        tildeomegai = ctx.tildeomegai
        i_idx = ctx.i_idx
        j_idx = ctx.j_idx
        sin2 = ctx.sin2
        X = ctx.X
        f = ctx.f
        xi = ctx.xi        
        nu = ctx.nu
        G = ctx.G
        F = ctx.F
            
        tu = c*(wt[None,:]*tau[None,:]/(tau[None,:]-u[:,None])).sum(axis=1)
        theta = c*(tau[None,:]*wt[None,:]/(tau[None,:]*X[:,None]+1)).sum(axis=1)
        
        if weights == "ewma":
            alpha = w_args[0]
            mldmp = mld_expp(mldm,alpha)
            integral1 = (mld_exp(xi/theta,alpha)+xi/theta*mld_expp(xi/theta,alpha))/theta**2
            integral3 = mld_expp(xi/theta,alpha)/theta**2
            
        elif weights == "unif":
            alpha = w_args[0]
            mldmp = mld_unifp(mldm,alpha)
            integral1 = (mld_unif(xi/theta,alpha)+xi/theta*mld_unifp(xi/theta,alpha))/theta**2
            integral3 = mld_unifp(xi/theta,alpha)/theta**2
            
        elif weights == "Ndiracs":
            mldmp = mld_Ndiracsp(mldm, wd, d)
            integral1 = (mld_Ndiracs(xi/theta, wd, d)+xi/theta*mld_Ndiracsp(xi/theta, wd, d))/theta**2
            integral3 = mld_Ndiracsp(xi/theta, wd, d)/theta**2
            
        else:
            mldmp = (wd[None,:]*d[None,:]/(d[None,:]-mldm[:,None])**2).sum(axis=1)
            integral1 = (wd[None,:]*d[None,:]**2/(xi[:,None] - d[None,:]*theta[None,:])**2).sum(axis=1)
            integral3 = (wd[None,:]*d[None,:]/(xi[:,None] - d[None,:]*theta[None,:])**2).sum(axis=1)
                
        dsidtk = c*u[:,None]**2*wt[None,:]/(tau[None,:] - u[:,None])**2*(mldm[:,None] + tu[:,None]/mldmp[:,None])
        
        dxiidtk = (1-sin2[:,None])*dsidtk[i_idx,:] + sin2[:,None]*dsidtk[i_idx+1,:]
        
        integral2 = c*(wt[None,:]*tau[None,:]**2/(tau[None,:]*X[:,None]+1)**2).sum(axis=1)
        dhatGammaidtk = c*wt[None,:]/(tau[None,:]*X[:,None]+1)**2*integral1[:,None]
        dhatGammaidXi = 1 - integral2*integral1
        dhatGammaidxii = -integral3
        dhatXidtk = - dhatGammaidtk/dhatGammaidXi[:,None]
        dXidxii = - dhatGammaidxii/dhatGammaidXi
        dXidtk = dhatXidtk + dXidxii[:,None]*dxiidtk
        
        dXidtk[torch.logical_not(torch.isfinite(dXidtk))] = 0
        
        integral4 = (tau[None,:]*wt[None,:]/(tau[None,:]*X[:,None]+1)**2).sum(axis=1)
        integral5 = (wt[None,:]/(tau[None,:]*X[:,None]+1)).sum(axis=1)
        dtildefidXi = integral4/xi
        dtildefidxii = integral5/xi**2
        dtildefidtk = wt[None,:]*X[:,None]/(tau[None,:]*X[:,None]+1)**2/xi[:,None]
        dfidtk = (dtildefidXi[:,None]*dXidtk + dtildefidxii[:,None]*dxiidtk + dtildefidtk).imag/np.pi
        
        dfidtk[torch.logical_not(torch.isfinite(dfidtk))] = 0
        
        dGidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        dFidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        for i in range(nu):
            j = torch.ones(omegai[i]+1, dtype=int).cumsum(0)
            offset = omegai[:i].sum(axis=0) + 2*i
            dGidtk[j+offset] =  1/2*((dxiidtk[j+offset,:] - dxiidtk[j-1+offset,:])*(f[j+offset,None] + f[j-1+offset,None])).cumsum(axis=0) + 1/2*((xi[j+offset,None] - xi[j-1+offset,None])*(dfidtk[j+offset,:] + dfidtk[j-1+offset,:])).cumsum(axis=0)
            if tildeomegai.sum(axis=0) > 0:
                if (G[offset+j][-1] - F[offset]).to(float) > 0:
                    dFidtk[j+offset] = tildeomegai[i]*(1-F[0])/(G[offset+j[-1]] - F[offset])*dGidtk[j+offset,:] - tildeomegai[i]*(1-F[0])/(G[offset+j[-1]] - F[offset])**2*(G[offset+j,None] - F[offset])*dGidtk[offset+j[-1:],:]
                else:
                    dFidtk[j+offset] = 0
                dFidtk[j[-1]+offset] = 0
            else:
                dFidtk[j+offset] = dGidtk[j+offset]
        if tildeomegai.sum(axis=0) == 0:
            dFidtk = dGidtk/G[-1] - G[:,None]/G[-1]**2*dGidtk[-1:,:]
        
        domegaidtk = torch.zeros((omegai.shape[0]+1, wt.shape[0]))
        doutidtk = torch.concatenate([dfidtk, dFidtk, dxiidtk, domegaidtk], axis=0)
        
        # For Gradient debug (test 4)
        # k = 2
        # print("dsi:", dsidtk[:,k])
        # print("dxi:", dxiidtk[:,k])
        # print("dXi:", dXidtk[:,k])
        # print("dfi:", dfidtk[:,k])
        # print("dFi:", dFidtk[:,k])
        # doutidtk = torch.ones(input.shape)[None,:]
        # print(grad_output[:,None] * doutidtk)
        
        # if (1-torch.isfinite(doutidtk).to(int)).sum() != 0:
        #     print("NaN grad")
        #     print(dsidtk)
        #     print(dxiidtk)
        #     print(integral1)
        #     print(integral2)
        #     print(dhatGammaidXi)
        #     print(dXidtk)
        #     print(dfidtk)
        #     print(dFidtk)
        #     print(domegaidtk)
        
        return grad_output[:,None] * doutidtk, None, None, None, None, None, None, None, None, None, None

def m_idx(K, d):
    M = d.shape[0]
    if K is None or K <= 0:
        return list(range(1,M+1))
    d_diff = torch.diff(d, axis=0)
    d_idx = torch.argsort(d_diff, axis=0)
    if K >= 2:
        rst = (d_idx[-K+1:]+1).tolist() + [M]
    else:
        rst = [M]
    return rst

def test1():
    dtype = torch.float
    device = torch.device("cpu")
    F = f_xi.apply
    input = torch.ones(3, device=device, dtype=dtype, requires_grad=True).cumsum(0)
    wt = None
    d = torch.ones(1)
    wd = torch.ones(1)
    c = 0.1
    mu = 1 
    weights = 'test'
    w_args = None
    omega = 2
    verbose = False
    output = F(input, wt, d, wd, c, mu, weights, w_args, omega, verbose) 
    loss = (output**2).sum()
    return output, loss

def test2():
    dtype = torch.float
    device = torch.device("cpu")
    F = f_xi.apply
    input = torch.ones(1, device=device, dtype=dtype, requires_grad=True)
    wt = None
    d = None
    wd = None
    c = 5
    mu = 1 
    weights = 'ewma'
    w_args = [0.01]
    omega = 200
    verbose = False
    output = F(input, wt, d, wd, c, mu, weights, w_args, omega, verbose) 
    nu = output[-1].to(int)
    omegai = output[-nu-1:-1]
    output = output[:-nu-1]
    f = output[:output.shape[0]//3]
    F = output[output.shape[0]//3:2*output.shape[0]//3]
    xi = output[2*output.shape[0]//3:]
    plt.figure()
    plt.plot(xi.detach().numpy(), f.detach().numpy(), label="Sample density")
    plt.plot(np.ones(2)*input.detach().numpy()[0], np.array([0,np.max(f.detach().numpy())]), label="Population density")
    plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
    plt.show()
    loss = (output**2).sum()
    return output, loss

def test3():
    dtype = torch.float
    device = torch.device("cpu")
    F = f_xi.apply
    
    input = torch.tensor([1,3,10], device=device, dtype=dtype, requires_grad=True)
    t = input
    wt = torch.tensor([0.2,0.4,0.4], device=device, dtype=dtype)
    Kt = 1
    
    c = 0.1
    mu = 0.
    alpha = 0.1
    weights = 'Ndiracs'   
    method = 'root'
    Kd = 1
    
    Nd = 100
    d = torch.tensor([0.5,81/2])
    wd = torch.tensor([1-1/80,1/80])
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    print("Method", method,", Nd =", Nd)
    
    omega = 100
    verbose = False
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    beg = time.process_time()
    output = F(input, wt, d, wd, c, mu, weights, w_args, omega, method, verbose)  
    end = time.process_time()
    print(end-beg, "s compute time.")
    nu = output[-1].to(int)
    omegai = output[-nu-1:-1]
    output = output[:-nu-1]
    f = output[:output.shape[0]//3]
    F = output[output.shape[0]//3:2*output.shape[0]//3]
    xi = output[2*output.shape[0]//3:]
    plt.figure()
    plt.plot(xi.detach().numpy(), f.detach().numpy(), label="Sample density")
    plt.hist(input.detach().numpy(), weights=wt.detach().numpy()*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=200, label="Population density")
    plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
    plt.show()
    
    
    plt.figure()
    plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Sample cdf")
    plt.hist(input.detach().numpy(), weights=wt.detach().numpy(), bins=200, cumulative = True, histtype="step", label="Population cdf")
    plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
    plt.show()
    
    loss = (output**2).sum()
    return input, output, loss

def test4():
    torch.set_printoptions(profile='default')
    torch.set_printoptions(linewidth=5000)
    dtype = torch.cfloat
    device = torch.device("cpu")
    F = f_xi.apply
    
    input = torch.tensor([1,3,10], device=device, dtype=dtype, requires_grad=True)
    t = input
    wt = None
    Kt = 1
    
    c = 0.1
    mu = 0.
    alpha = 0.1
    weights = 'Ndiracs'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    d = torch.tensor([0.5,81/2])
    wd = torch.tensor([1-1/80,1/80])
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    print("Method", method,", Nd =", Nd)
    
    omega = 5
    verbose = False
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    
    #output = F(input, wt, d, wd, c, mu, weights, w_args, omega, verbose) 
    output = F(input, wt, d, wd, c, mu, weights, w_args, omega, method, verbose)
    loss = torch.abs(output).sum()
    output.retain_grad()
    loss.backward()
    
    eps = 1e-3
    input2 = input + eps*torch.tensor([0,0,1], dtype=dtype, requires_grad=True)
    output2 = F(input2, wt, d, wd, c, mu, weights, w_args, omega, method, verbose)
    print()
    k = 8
    print("dsi:", ((output2-output)/eps)[4*k:])
    print("dxi:", ((output2-output)/eps)[2*k:3*k])
    print("dXi:", ((output2-output)/eps)[3*k:4*k])
    print("dfi:", ((output2-output)/eps)[:k])
    print("dFi:", ((output2-output)/eps)[k:2*k])
    
    print()
    print(output[k:2*k])
    print(output2[k:2*k])
    return output, output2, loss

if __name__ == '__main__':
    #output, loss = test1()
    #output, loss = test2()
    #input, output, loss = test3()
    output, output2, loss = test4()
