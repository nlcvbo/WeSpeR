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
from wQuEST import f_xi, m_idx
import time
from weighted_LWO_estimator import  wLWO_estimator

class wQuEST_model(torch.nn.Module):
    def __init__(self, tau_init, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, method = 'root', verbose = True):
        """
            Inputs:
                tau_init: float tensor of shape (p), initial guess of population spectrum
                d: float tensor of sahpe (n), weight vector (the diagonal of the weight matrix)
        """
        super().__init__()
        self.tau_mean = tau_init.clone().mean(axis=0)
        self.logtau = torch.nn.Parameter(torch.log(tau_init.clone()/self.tau_mean).type(torch.float64))
        self.wt = wt
        self.d = d
        self.wd = wd
        self.c = c
        self.mu = mu
        self.weights = weights
        self.w_args = w_args
        self.omega = omega
        self.method = method
        self.verbose = verbose
    
    def get_tau(self):
        return torch.exp(self.logtau)*self.tau_mean
        
    def forward(self):
        """
            Inputs:
            Outputs:
                
        """    
        F = f_xi.apply
        idx = torch.argsort(self.logtau)
        if self.wt is None:
            output = F(torch.exp(self.logtau[idx])*self.tau_mean, self.wt, self.d, self.wd, self.c, self.mu,self.weights, self.w_args, self.omega, self.method, self.verbose) 
        else:
            output = F(torch.exp(self.logtau[idx])*self.tau_mean, self.wt[idx], self.d, self.wd, self.c, self.mu,self.weights, self.w_args, self.omega, self.method, self.verbose) 
        return output


def loss_wasserstein_1D_cdf(p=2, regu1 = 1, regu2 = 1):
    def loss_wasserstein_p(u_values, v_values, u_cumul = None, v_cumul = None):
        # Inspired from Scipy 1D Wasserstein distance: https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807
        
        if u_cumul is None:
            u_cumul = torch.ones(u_values.shape).cumsum(axis=0)-1
            u_cumul = u_cumul / u_cumul[-1]
        if v_cumul is None:
            v_cumul = torch.ones(v_values.shape).cumsum(axis=0)-1
            v_cumul = v_cumul / v_cumul[-1]
        
        u_weights = torch.diff(torch.cat((u_cumul[:1],u_cumul)))
        v_weights = torch.diff(torch.cat((v_cumul[:1],v_cumul)))
        
        u_sorter = torch.argsort(u_values, axis=0)
        v_sorter = torch.argsort(v_values, axis=0)
        
        all_values = torch.cat((u_values, v_values), axis=0)
        all_values = all_values.sort(axis=0)[0]
        
        # Compute the differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, axis=0)
        
        # Get the respective positions of the values of u and v among the values of both distributions.
        u_values_sort = u_values[u_sorter].contiguous()
        v_values_sort = v_values[v_sorter].contiguous()
        
        u_cdf_indices = torch.searchsorted(all_values[:-1].contiguous(), u_values_sort, right = True)
        v_cdf_indices = torch.searchsorted(all_values[:-1].contiguous(), v_values_sort, right = True)
        
        # Calculate the CDFs of u and v using their weights, if specified.
        u_cdf = torch.zeros(all_values.shape, dtype=torch.float)
        v_cdf = torch.zeros(all_values.shape, dtype=torch.float)
        
        u_cdf[u_cdf_indices] = u_weights[u_sorter].to(torch.float)
        u_cdf = u_cdf.cumsum(axis=0)
        
        v_cdf[v_cdf_indices] = v_weights[v_sorter].to(torch.float)
        v_cdf = v_cdf.cumsum(axis=0)
        
        
        # We do not normalize the power by 1/p at the end, to make it differentiable
        distance1 = ((torch.abs(u_cdf - v_cdf)[:-1]**p * deltas)).sum(axis=0)
        
        distance2 = regu1*((u_values*u_weights).sum(axis=0) - (v_values*v_weights).sum(axis=0))**2 
        
        if u_values.shape[0] > 1:
            u_max = torch.max(u_values[u_weights > 0])
            u_min = torch.min(u_values[u_weights > 0])
            q_max = u_cumul[-2]
            q_min = u_cumul[2]
            v_max_idx = v_cumul >= q_max
            v_min_idx = v_cumul <= q_min
            v_max = torch.max(v_values[v_weights > 0])
            v_min = torch.min(v_values[v_weights > 0])
            # if v_values[v_max_idx].shape[0] > 0:
            #     v_max = v_values[v_max_idx].mean()
            # else:
            #     v_max = v_values[-1]
            # if v_values[v_min_idx].shape[0] > 0:
            #     v_min = v_values[v_min_idx].mean()
            # else:
            #     v_min = v_values[0]
            
            distance3 = regu2*((v_min - u_min)**2 + (v_max - u_max)**2)
        else:
            distance3 = 0
        return (distance1 + distance2 + distance3).mean()
    
    return loss_wasserstein_p

def loss_LW(regu1 = 0., regu2 = 0.):
    def loss_fn(lambda_, F, xi, c, omegai):
        
        
        p = lambda_.shape[0]
        n = int(p/c)
        lambda_k = torch.zeros(p)
        X_k = 0
        X_kp = 0
        
        idx = torch.ones(F.shape[0], dtype=int).cumsum(axis=0) - 1
        omegai_cum = omegai.cumsum(axis=0)
        for kappa in range(max(p-n,0), p):
            kappap = kappa + 1
            
            j_idx = idx[torch.logical_and(F >= kappa/p, F < kappap/p)]
            
            if j_idx.shape[0] == 0:
                j_kp = max(torch.argmax(((kappap - p*F) < 0).to(int)) - 1, 0)
                j_k = max(torch.argmax(((kappap - p*F) < 0).to(int)) - 1, 0)
                i = torch.argmax(((j_k - omegai_cum) < 0).to(int))                
                ip = torch.argmax(((j_kp - omegai_cum) < 0).to(int))
                delta1 = (kappap/p - F[j_kp])*xi[j_kp] + (kappap/p - F[j_kp])**2/2/(F[j_kp+1] - F[j_kp])*(xi[j_kp+1] - xi[j_kp])
                delta2 =  -(kappa/p - F[j_k])*xi[j_k] - (kappa/p - F[j_k])**2/2/(F[j_k+1] - F[j_k])*(xi[j_k+1] - xi[j_k])
                delta3 = torch.zeros(1)
                
            else:
                j_kp = j_idx[-1]
                ip = torch.argmax(((j_kp - omegai_cum) < 0).to(int))
            
                j_k = j_idx[0]
                i = torch.argmax(((j_k - omegai_cum) < 0).to(int))
            
                if j_kp+1 < F.shape[0] and F[j_kp+1] - F[j_kp] > 0:
                    delta1 = (kappap/p - F[j_kp])*xi[j_kp] + (kappap/p - F[j_kp])**2/2/(F[j_kp+1] - F[j_kp])*(xi[j_kp+1] - xi[j_kp])
                else:
                    delta1 = torch.zeros(1)
                    
                if j_k-1 >= 0 and F[j_k] - F[j_k-1] > 0:
                    delta2 = (F[j_k] - kappa/p)*xi[j_k] - (F[j_k] - kappa/p)**2/2/(F[j_k] - F[j_k-1])*(xi[j_k] - xi[j_k-1])
                else:
                    delta2 = torch.zeros(1)
                
                if j_kp > j_k:
                    l = torch.ones(j_kp - j_k, dtype=int).cumsum(axis=0) - 1 + j_k
                    delta3 = ((F[l+1] - F[l])*(xi[l+1] + xi[l])/2).sum(axis=0)
                else:
                    delta3 = torch.zeros(1)
                
                    
            
            X_k = X_kp
            X_kp = X_k + delta1 + delta2 + delta3
            
            if kappa > 0:        
                X_kp = max(X_k + lambda_k[kappa-1]/p, X_kp)
            lambda_k[kappa] = (X_kp - X_k)*p
            
            if i != ip:
                j = torch.argmin((lambda_k[kappa] - xi)**2)
                lambda_k[kappa] = xi[j]
            
        # first penalization on the mean
        pen1 = regu1*(lambda_.mean(axis=0) - lambda_k.mean(axis=0))**2
        # second penalization on the extremums
        idx = torch.ones(lambda_.shape, dtype=int).cumsum(axis=0)-1
        weights = torch.cos(idx*np.pi/idx[-1])**2
        pen2 = regu2*(weights*(lambda_ - lambda_k)**2).mean()
            
            
        return (((lambda_ - lambda_k)**2).mean() + pen1 + pen2)/(1 + regu1 + regu2)
    return loss_fn

def loss_LWO():
    def loss_fn(lambda_, F, xi, c):
        p = lambda_.shape[0]
        n = int(p/c)
        lambda_k = torch.zeros(p)
        for kappa in range(max(p-n,0), p):
            j_k = max(torch.argmax((((kappa+1/2) - p*F) < 0).to(int)) - 1, 0)
            j_kp = max(torch.argmax((((kappa+1/2) - p*F) < 0).to(int)), 0)
            if F[j_kp] - F[j_k] > 0:
                lambda_k[kappa] = (((kappa+1/2)/p - F[j_k])*xi[j_k] + (F[j_kp] - (kappa+1/2)/p)*xi[j_kp])/(F[j_kp] - F[j_k])
            else:
                lambda_k[kappa] = (xi[j_k]+xi[j_kp])/2
        
        return ((lambda_ - lambda_k)**2).mean()
    return loss_fn

def wQuEST(lambda_, tau_init, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, lr = 1e-2, n_epochs = 100, decay = 0.97, loss_type="w2", method = 'root', verbose = True):   
    model = wQuEST_model(tau_init, wt = wt, d = d, wd = wd, c =c, mu = mu, weights = weights, w_args = w_args, omega = omega, method = method, verbose = False)
    if loss_type == "w2":
        loss_fn = loss_wasserstein_1D_cdf(p=2, regu1=0.1, regu2=0.1) # we can also use loss_composite(p=2, regu=1)
    elif loss_type == "trap":
        loss_fn = loss_LW(regu1=0.1, regu2=1.)
    elif loss_type == "euler":
        loss_fn = loss_LWO()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    
    # Train loop
    all_tau = [model.get_tau().detach()]
    running_loss = []
    best_loss = np.inf
    best_tau = tau_init.type(torch.float64)
    
    eps = 1e-5
    
    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        output = model()
            
        nu = output[-1].to(int)
        omegai = output[-nu-1:-1].to(int)
        output = output[:-nu-1]
        f = output[:output.shape[0]//3]
        F = output[output.shape[0]//3:2*output.shape[0]//3]
        xi = output[2*output.shape[0]//3:]
        u_cumul = None
        v_cumul = F
        if loss_type == "w2":
            loss = loss_fn(torch.cat([torch.zeros(1),lambda_]), xi, u_cumul, v_cumul)
        elif loss_type == "trap":
            loss = loss_fn(lambda_, F, xi, c, omegai)
        elif loss_type == "euler":
            loss = loss_fn(lambda_, F, xi, c)
        loss.backward()
            
        optimizer.step()
        scheduler.step()
        
        # for p in model.parameters():
        #     p.data.clamp_(eps, None)
        #     if c < 1:
        #         p.data.clamp_(torch.min(lambda_), torch.max(lambda_))
        
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = model.get_tau().clone()
            
        all_tau += [model.get_tau().detach()]
        running_loss += [loss.item()]
        if verbose:
            if n_epochs < 11 or i % min((n_epochs // 10),10) == 0:
                print("Loss epoch", i, ":", loss.item())
        
    if verbose:
        print("Final loss :", best_loss)
        plt.figure()
        plt.semilogy(np.array(running_loss))
        plt.title("Running loss in function of epochs")
        plt.xlabel("epochs")
        plt.ylabel("training loss")
        plt.show()
    
    return model, best_tau, all_tau
