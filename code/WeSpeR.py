import torch
import scipy as sp
import numpy as np
from WeSpeR_support_identification_unif import find_support_unif
from WeSpeR_support_identification_ewma import find_support_exp
from WeSpeR_support_identification_Ndiracs import find_support_Ndiracs
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import time


def m_tilde_loss(ab, x, w, c, t, wt):
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    u = 1 + c*w[None,:]*(wt[None,:]*(t[None,:]*a[:,None]-x[:,None])*t[None,:]/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-np.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    v = - c*w[None,:]*(wt[None,:]*t[None,:]**2*(-np.exp(b)[:,None])/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-np.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    loss = (a - (w[None,:]*u/(u**2+v**2)).mean(axis=1))**2 + (-np.exp(b) + (w[None,:]*v/(u**2+v**2)).mean(axis=1))**2
    return loss.mean()

def m(x, w, c, t, wt):
    m_tilde = np.zeros(x.shape, np.complex64)
    for i in range(x.shape[0]):
        m = minimize(m_tilde_loss, x0 = np.array([0,0]), args = (x[i:i+1], w, c, t, wt), options={'disp':False})
        if m.fun < 1e-4:
            ab = m.x
        else:
            ab = np.array([0,0])
        m_tilde[i] = ab[0] - np.exp(ab[1])*1j
    m = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None])).sum(axis=1)
    return m

def MP_pdf(x, w, c, t, wt):
    return np.abs(m(x, w, c, t, wt).imag)/np.pi

class WeSpeR_model(torch.nn.Module):
    def __init__(self, tau_init, W, p, n, b):
        """
            Inputs:
                tau_init: float tensor of shape (p), initial guess of population spectrum
                W: float tensor of sahpe (n), weight vector (the diagonal of the weight matrix)
                p: int, dimension
                n: int, number of samples
                b: int, batch size
        """
        super().__init__()
        self.tau = torch.nn.Parameter(tau_init.clone().type(torch.float64))
        self.p = p
        self.n = n
        self.b = b
        self.Wsq = torch.sqrt(W)
        
    def forward(self):
        """
            Inputs:
            Outputs:
                lambda_: float tensor of shape (self.b, self.p), sample eigenvalues
        """    
        X = torch.empty(size = (self.b,self.n,self.p), dtype = torch.float64).normal_()*self.Wsq[None,:,None]
        sq_cov = torch.diag(torch.sqrt(torch.abs(self.tau).sort()[0]))[None,:,:]
        S = sq_cov @ X.transpose(-2,-1) @ X @ sq_cov/self.n
        # torch.eigvals segfault when there are NaNs
        if S.isnan().sum() > 0:
            raise ValueError("NaN not supported in torch.eigvals")
        S[S.isnan()] = 0
        lambda_ = torch.linalg.eigvals(S).real
        lambda_ = lambda_.sort()[0]
        return lambda_

def loss_wasserstein_1D(p=2, regu = 1):
    def loss_wasserstein_p(u_values, v_values, u_weights = None, v_weights = None):
        # Inspired from Scipy 1D Wasserstein distance: https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807
        if u_weights is None:
            u_weights = torch.ones(u_values.shape)
            u_weights = u_weights / u_weights.sum(axis = 1)[:,None]
        if v_weights is None:
            v_weights = torch.ones(v_values.shape)
            v_weights = v_weights / v_weights.sum(axis = 1)[:,None]
        
        u_sorter = torch.argsort(u_values, axis=1)
        u_sorter = (torch.ones(u_values.shape, dtype=int).cumsum(0)-1, u_sorter)
        v_sorter = torch.argsort(v_values, axis=1)
        v_sorter = (torch.ones(v_values.shape, dtype=int).cumsum(0)-1, v_sorter)
        
        all_values = torch.cat((u_values, v_values), axis=1)
        all_values = all_values.sort(axis=1)[0]
    
        # Compute the differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, axis=1)
    
        # Get the respective positions of the values of u and v among the values of both distributions.
        u_values_sort = u_values[u_sorter].contiguous()
        v_values_sort = v_values[v_sorter].contiguous()
        
        u_cdf_indices = torch.searchsorted(all_values[:,:-1].contiguous(), u_values_sort, right = True)
        u_cdf_indices = (torch.ones(u_cdf_indices.shape, dtype=int).cumsum(0)-1, u_cdf_indices)
        v_cdf_indices = torch.searchsorted(all_values[:,:-1].contiguous(), v_values_sort, right = True)
        v_cdf_indices = (torch.ones(v_cdf_indices.shape, dtype=int).cumsum(0)-1, v_cdf_indices)
        
        # Calculate the CDFs of u and v using their weights, if specified.
        u_cdf = torch.zeros(all_values.shape)
        v_cdf = torch.zeros(all_values.shape)
        
        u_cdf[u_cdf_indices] = u_weights[u_sorter]
        u_cdf = u_cdf[:,1:].cumsum(axis=1)
        u_cdf = u_cdf / u_cdf[:,-1:]
        
        v_cdf[v_cdf_indices] = v_weights[v_sorter]
        v_cdf = v_cdf[:,1:].cumsum(axis=1)
        v_cdf = v_cdf / v_cdf[:,-1:]
        
        # We do not normalize the power by 1/p at the end, to make it differentiable
        distance = ((torch.abs(u_cdf - v_cdf)**p * deltas)).sum(axis=1) + regu*(u_values.mean(axis=1) - v_values.mean(axis=1))**2
        return distance.mean()
    
    return loss_wasserstein_p

def loss_composite(p=2, regu=1):
    def loss(u_values, v_values):
        return loss_wasserstein_1D(p, regu)(u_values, v_values) + ((u_values - v_values)**2).mean()
    return loss

def WeSpeR_minimization(train_lambda_, W, tau_init, n_epochs = 200, b = 64, lr = 5e-2, verbose = True):
    """
        Inputs:
            train_lambda_: float torch tensor of shape (p), sample eigenvalues
            W: float torch tensor of sahpe (n), weight vector (the diagonal of the weight matrix) 
            tau_init: float torch tensor of shape (p), initial guess of population spectrum
            b: int, batch size
            n_epochs: int, max number of iterations in the optimizer
            lr: float, learning rate for Adam
            verbose: bool, if True, will print information
        Outputs:
            model: torch WeSpeR_model
    """
    p = train_lambda_.shape[0]
    n = W.shape[0]
    
    model = WeSpeR_model(tau_init, W, p, n, b)
    loss_fn = loss_wasserstein_1D(p=2, regu=1) # we can also use loss_composite(p=2, regu=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train loop
    running_loss = []
    best_loss = np.inf
    best_tau = tau_init.type(torch.float64)
    train_lambda_batch = train_lambda_.tile((b,1))
    
    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        lambda_ = model()
        loss = loss_fn(lambda_, train_lambda_batch)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = model.tau
            
        running_loss += [loss.item()]
        if verbose:
            if n_epochs < 11 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
        
    if verbose:
        print("Final loss :", best_loss)
        plt.figure()
        plt.plot(np.array(running_loss))
        plt.title("Running loss in function of epochs")
        plt.xlabel("epochs")
        plt.ylabel("training loss")
        plt.show()
    
    return model, best_tau

def WeSpeR_step1(X_sample, W, tau_init = None, assume_centered = False, verbose = True):
    """
    Inputs:
        X_sample: float torch tensor of shape (p), data matrix
        W: float torch tensor of shape (n), weight vector (the diagonal of the weight matrix) 
        tau_init: float torch tensor of shape (p) or None, initial guess of population spectrum
        assume_centered: bool, if False, X_sample will be demeaned
        verbose: bool, if True, will print information

    Outputs:
        train_lambda_: float torch tensor of shape (p), sample eigenvalues
        tau_init: float torch tensor of shape (p), initial guess of population spectrum

    """
    n, p = X_sample.shape
    
    Y_sample = X_sample*torch.sqrt(W)[:,None]
    if assume_centered:
        sample_cov = Y_sample.T @ Y_sample/n
    else:
        Y_mean = Y_sample.mean(axis=0)[None,:]
        sample_cov = (Y_sample - Y_mean).T @ (Y_sample - Y_mean)/(n-1)
    train_lambda_, V = torch.linalg.eigh(sample_cov)
    
    try:
        assert(tau_init.shape == (p,p))
    except:
        LW_cov = LedoitWolf().fit(Y_sample.numpy()).covariance_
        tau_init, _ = sp.linalg.eigh(LW_cov)
        tau_init = torch.tensor(tau_init)
    
    return train_lambda_, tau_init

def WeSpeR_step2(train_lambda_, W, tau_init, n_epochs = 200, b = 64, lr = 5e-2, verbose = True):
    """
    Inputs:
        train_lambda_: float torch tensor of shape (p), sample eigenvalues
        W: float torch tensor of shape (n), weight vector (the diagonal of the weight matrix) 
        tau_init: float torch tensor of shape (p), initial guess of population spectrum
        b: int, batch size
        n_epochs: int, max number of iterations in the optimizer
        lr: float, learning rate for Adam
        verbose: bool, if True, will print information

    Outputs:
        support : numpy array of shape (2*nu), increasing order

    """
    model, best_tau = WeSpeR_minimization(train_lambda_, W, tau_init, n_epochs = n_epochs, b = b, lr = lr, verbose = verbose)
    tau = np.sort((best_tau.detach().numpy()).real)
    return tau

def WeSpeR_step3(tau, W, c, weights, w_args, verbose = True):
    t = np.unique(tau)
    w = np.ones(tau.shape[0])/tau.shape[0]
    
    if weights == "ewma":
        alpha = w_args[0]
        support = find_support_exp(c, alpha, t, w, verbose = verbose)
        
    elif weights == "unif":
        alpha = w_args[0]
        support = find_support_unif(c, alpha, t, w, verbose = verbose)
        
    elif weights == "Ndiracs":
        if w_args == None or len(w_args) <= 1:
            d = W.numpy()
            wd = np.ones(d.shape[0])/d.shape[0]
        else:
            d = w_args[0]
            wd = w_args[1]
        support = find_support_Ndiracs(c, wd, d, t, w, verbose = verbose)
    return np.array(support)

def WeSpeR_step4(omega, support, W, c, tau, train_lambda_ = None, mu = 0.1, wt = None, verbose = True):
    assert support.shape[0] % 2 == 0
    if support.shape[0] == 0:
        try:
            support = np.array([np.min(train_lambda_)/1.2, np.max(train_lambda_)*1.2])
        except:
            support = np.array([np.min(tau)*(max(1-np.sqrt(c),1e-2))**2, np.max(tau)*(1+np.sqrt(c))**2])
    nu = support.shape[0]//2
    
    try: p=train_lambda_.shape[0]
    except: mu = 1
    mu = max(0,min(mu,1))
    
    # Number of points
    omegai = np.ones(nu)*mu*omega/nu
    if mu < 1:
        for i in range(nu):
            ul = support[2*i]
            ur = support[2*i+1]
            card = ((train_lambda_>=ul)*(train_lambda_<=ur)).sum()
            omegai[i] += (1-mu)*card/p
    
    # Discretization
    if omegai.sum() > 0:
        omegai = np.around(omegai*omega/omegai.sum(),0).astype(int)
    
    # Grid
    grid = np.zeros(0)
    density = np.zeros(0)
    for i in range(nu):
        ul = support[2*i]
        ur = support[2*i+1]
        j = np.ones(omegai[i]).cumsum()
        xij = ul + (ur-ul)*np.sin(np.pi*j/2/(omegai[i]+1))**2
        try:
            assert wt.shape == tau.shape
            f = MP_pdf(xij, W.numpy(), c, tau, wt)
        except:
            f = MP_pdf(xij, W.numpy(), c, tau, np.ones(tau.shape[0])/tau.shape[0])
        grid = np.concatenate([grid, ul*np.ones(1), xij, ur*np.ones(1)])
        density = np.concatenate([density, np.zeros(1), f, np.zeros(1)])
    
    return grid, density

def WeSpeR(X_sample, W, omega, tau_init = None, n_epochs = 200, b = 64, mu = 0.1, assume_centered = False, lr = 5e-2, weights = "ewma", w_args = [1e0], verbose = True):
    """
        Inputs:
            X_sample: float torch tensor of shape (n, p), with n the number of samples, and p the dimension
            W: float torch tensor of shape (n), weight vector (the diagonal of the weight matrix) 
            omega: int, number of discretization points
            tau_init: float torch tensor of shape (p), initial guess of population spectrum
            b: int, batch size
            mu: float, hyperparameter for grid definition strategy
            assume_centered: bool, if False, X_sample will be demeaned
            n_epochs: int, max number of iterations in the optimizer
            lr: float, learning rate for Adam
            weights: str, "ewma", "unif" or "Ndiracs"
            w_args: list args for the weight distribution
            verbose: bool, if True, will print information
        Outputs:
            tau: numpy array of shape (p), the estimated population eigenvalues
            support: float np array of dimension 1, support boundaries of F
            grid: discretization grid for density estimation
            density: sample density estimation on the grid
    """
    n, p = X_sample.shape
    c = p/n
    
    train_lambda_, tau_init = WeSpeR_step1(X_sample, W, tau_init = tau_init, assume_centered = assume_centered, verbose = False)
    
    tau = WeSpeR_step2(train_lambda_, W, tau_init, n_epochs = n_epochs, b = b, lr = lr, verbose = verbose)
    
    support = WeSpeR_step3(tau, W, c, weights, w_args, verbose = False)
    
    grid, density = WeSpeR_step4(omega, support, W, c, tau, train_lambda_ = train_lambda_.numpy(), mu = mu, verbose = False)
    
    return tau, support, grid, density
    

def WeSpeR_experiment_0():
    # Basic experiment without fitting completely, to show that each part works
    n = 400
    p = 40
    alpha = 1
    n_epochs = 1
    lr = 5e-2
    b = 64
    mu = 0.1
    nu = None
    omega = 200
    verbose = True
    weights = "ewma"
    w_args = [alpha]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    beta = alpha/(1-np.exp(-alpha))
    W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    W = torch.tensor(W)
    
    # Sampling
    if nu == None:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
    else:
        X_sample = stats.t.rvs(nu, size=(n,p))*np.sqrt((nu-2)/nu) @ np.sqrt(np.diag(tau_pop)) 
    X_sample = torch.tensor(X_sample)
    
    start_time = time.time()
    tau, support, grid, density = WeSpeR(X_sample, W, omega, tau_init = None, n_epochs = n_epochs, b = b, mu = mu, assume_centered = True, lr = lr, weights = weights, w_args = w_args, verbose = False)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")
    
    c = p/n
    t = np.array([1.,3.,10.])
    w = np.array([0.2,0.4,0.4])
    
    support_pop = np.array(find_support_exp(c, alpha, t, w, verbose = False))
    train_lambda_, _ = WeSpeR_step1(X_sample, W, tau_init = None, assume_centered = True, verbose = False)
    grid_pop, density_pop = WeSpeR_step4(omega, support_pop, W, c, t, train_lambda_ = train_lambda_.numpy(), mu = 0.1, wt = w, verbose = False)
    
    
    plt.figure()
    plt.hist(train_lambda_, bins = list(np.linspace(0,17.5,70)), label=r"Sample eigenvalues", density=True, histtype='step')
    plt.plot(grid, density, label=r'Estimated sample density')
    plt.plot(grid_pop, density_pop, label=r'True sample density')#, linestyle=(0,(5,5)))
    plt.title("Weighted sample covariance eigenvalue pdf and histogram, p="+str(p))
    plt.legend()
    plt.show()

def WeSpeR_experiment_1(nu = None):
    n = 4000
    p = 400
    alpha = 1
    n_epochs = 120
    lr = 5e-2
    b = 256
    mu = 0.1
    omega = 2000
    nu = None
    verbose = True
    weights = "ewma"
    w_args = [alpha]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    beta = alpha/(1-np.exp(-alpha))
    W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    W = torch.tensor(W)
    
    # Sampling
    if nu == None:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
    else:
        X_sample = stats.t.rvs(nu, size=(n,p))*np.sqrt((nu-2)/nu) @ np.sqrt(np.diag(tau_pop))  
    X_sample = torch.tensor(X_sample)
    
    start_time = time.time()
    tau, support, grid, density = WeSpeR(X_sample, W, omega, tau_init = None, n_epochs = n_epochs, b = b, mu = mu, assume_centered = True, lr = lr, weights = weights, w_args = w_args, verbose = verbose)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")
    
    c = p/n
    t = np.array([1.,3.,10.])
    w = np.array([0.2,0.4,0.4])
    
    support_pop = np.array(find_support_exp(c, alpha, t, w, verbose = False))
    train_lambda_, _ = WeSpeR_step1(X_sample, W, tau_init = None, assume_centered = True, verbose = False)
    grid_pop, density_pop = WeSpeR_step4(omega, support_pop, W, c, t, train_lambda_ = train_lambda_.numpy(), mu = 0.1, wt = w, verbose = False)
    
    
    plt.figure()
    plt.hist(train_lambda_, bins = list(np.linspace(0,17.5,70)), label=r"Sample eigenvalues", density=True, histtype='step')
    plt.plot(grid, density, label=r'Estimated sample density')
    plt.plot(grid_pop, density_pop, label=r'True sample density', linestyle=(0,(5,5)))
    plt.title("Weighted sample covariance eigenvalue pdf and histogram, p="+str(p))
    plt.legend()
    plt.show()

def WeSpeR_experiment_2(nu = None):
    n = 800
    p = 400
    alpha = 1
    n_epochs = 120
    lr = 5e-2
    b = 256
    mu = 0.1
    omega = 2000
    nu = None
    verbose = True
    weights = "ewma"
    w_args = [alpha]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    beta = alpha/(1-np.exp(-alpha))
    W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    W = torch.tensor(W)
    
    # Sampling
    if nu == None:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
    else:
        X_sample = stats.t.rvs(nu, size=(n,p))*np.sqrt((nu-2)/nu) @ np.sqrt(np.diag(tau_pop))  
    X_sample = torch.tensor(X_sample)
    
    start_time = time.time()
    tau, support, grid, density = WeSpeR(X_sample, W, omega, tau_init = None, n_epochs = n_epochs, b = b, mu = mu, assume_centered = True, lr = lr, weights = weights, w_args = w_args, verbose = verbose)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")
    
    c = p/n
    t = np.array([1.,3.,10.])
    w = np.array([0.2,0.4,0.4])
    
    support_pop = np.array(find_support_exp(c, alpha, t, w, verbose = False))
    train_lambda_, _ = WeSpeR_step1(X_sample, W, tau_init = None, assume_centered = True, verbose = False)
    grid_pop, density_pop = WeSpeR_step4(omega, support_pop, W, c, t, train_lambda_ = train_lambda_.numpy(), mu = 0.1, wt = w, verbose = False)
    
    
    plt.figure()
    plt.hist(train_lambda_, bins = list(np.linspace(0,17.5,70)), label=r"Sample eigenvalues", density=True, histtype='step')
    plt.plot(grid, density, label=r'Estimated sample density')
    plt.plot(grid_pop, density_pop, label=r'True sample density', linestyle=(0,(5,5)))
    plt.title("Weighted sample covariance eigenvalue pdf and histogram, p="+str(p))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Test experiment, no fitting (1 epoch) to check if all works correctly
    WeSpeR_experiment_0()
    # Full fit below, can be slow to ensure convergence is completed
    WeSpeR_experiment_1(nu = None)
    WeSpeR_experiment_1(nu = 4)
    WeSpeR_experiment_1(nu = 3)
    WeSpeR_experiment_2(nu = None)