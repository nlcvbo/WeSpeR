import torch
import scipy as sp
import numpy as np
import scipy.stats as stats

class auto_QuEST(torch.nn.Module):
    def __init__(self, tau_init, Wsq, p, n, b):
        super().__init__()
        self.logtau = torch.nn.Parameter(torch.log(tau_init).clone().type(torch.float64))
        self.p = p
        self.n = n
        self.b = b
        self.Wsq = Wsq
    
    def get_tau(self):
        return torch.exp(self.logtau)
        
    def forward(self):
        """
            Inputs:
            Outputs:
                lambda_: float tensor of shape (self.b, self.p), sample eigenvalues
        """    
        
        if self.n > self.p:
            n_left = self.n
            n_used = 0
            S = torch.zeros(size = (self.b,self.p,self.p), dtype = torch.float64)
            sq_cov = torch.diag(torch.sqrt(torch.exp(self.logtau).sort()[0]))[None,:,:]
            while n_left > 0:
                n_sample = min(n_left, self.p)
                X_sample = torch.empty(size = (self.b,n_sample,self.p), dtype = torch.float64).normal_()*self.Wsq[None,n_used:n_used+n_sample,None] 
                S += sq_cov @ X_sample.transpose(-2,-1) @ X_sample @ sq_cov/self.n
                n_used += n_sample
                n_left -= n_sample
        else:
            X = torch.empty(size = (self.b,self.n,self.p), dtype = torch.float64).normal_()*self.Wsq[None,:,None]
            sq_cov = torch.diag(torch.sqrt(torch.exp(self.logtau).sort()[0]))[None,:,:]
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
    
        # Get the respective positions of the values of u and v among the values of
        # both distributions.
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
    
def auto_QuEST_minimization(S, n, p, tau_init, Wsq, method = "Adam", n_epochs = 1000, b = 64, assume_centered = False, lr = 1e-2, momentum=0., verbose = True):
    """
        Inputs:
            X_sample: float tensor of shape (n, p), with n the number of samples, and p the dimension
            Wsq: W^{1/2}
            tau_init: float tensor of shape (p), initial guess of population spectrum
            b: int, batch size
            assume_centered: Bool, if False, X_sample will be demeaned
            max_iter: int, max number of iterations in the optimizer
            lr: float, learning rate for GD and Adam
        Outputs:
            tau: tensor of shape (p), the estimated population eigenvalues
    """
    
    train_lambda_, V = torch.linalg.eigh(S)
    
    # Optimizers specified in the torch.optim package
    model = auto_QuEST(tau_init, Wsq, p, n, b)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = loss_wasserstein_1D(p=2, regu=1)
    loss_fn = loss_composite(p=2, regu=1)
    if method == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif method == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train loop
    running_loss = []
    best_loss = np.inf
    best_tau = tau_init.type(torch.float64)
    all_tau = [model.get_tau().detach()]
    train_lambda_batch = train_lambda_.tile((b,1))
    
    model.train(True)
    for i in range(n_epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        lambda_ = model()

        # Compute the loss and its gradients
        loss = loss_fn(lambda_, train_lambda_batch)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = model.get_tau().detach()
        
        # Gather data and report
        running_loss += [loss.item()]
        all_tau += [model.get_tau().detach()]
        
        if verbose:
            if n_epochs < 10 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
        
    if verbose:
        print("Final loss :", best_loss)
    
    return model, best_tau, np.array(running_loss), all_tau

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.covariance import LedoitWolf
    from QuEST import QuEST, NL_covariance
    
    
    n = 4000
    p = 400
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    
    nu = 3
    
    alpha = 1
    beta = alpha/(1-np.exp(-alpha))
    W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    Wsq = torch.tensor(np.sqrt(W))
    
    # Minimization
    minim = True
    if minim:
        # Params
        max_iter = 40
        lr = 5e-2
        b = 256
        method = "Adam"
        
        # Sampling
        cov = np.diag(tau_pop)
        try:
            X_sample.shape
        except:
            if nu == None:
                X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
            else:
                X_sample = stats.t.rvs(nu, size=(n,p))*np.sqrt((nu-2)/nu) @ np.sqrt(np.diag(tau_pop))  
            X_sample = torch.tensor(X_sample)*Wsq[:,None]
        
        # Initialization
        LW_cov = LedoitWolf().fit(X_sample).covariance_
        try:
            tau_init = tau_quest
        except:
            tau_init, _ = sp.linalg.eigh(LW_cov)
        tau_init = torch.tensor(tau_init)
        
        # tau_init = torch.tensor(tau_quest)
        # Minimization
        start_time = time.time()
        
        model, best_tau, running_loss = auto_QuEST_minimization(X_sample, tau_init, Wsq, method = method, n_epochs = max_iter, b = b, assume_centered = True, lr = lr, momentum=0., verbose = True)
        tau_quest = np.sort(np.abs(best_tau.numpy()))
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")
        
        X = X_sample.numpy()
        lambda_, _ = np.linalg.eigh(X.T @ X/X.shape[0])
        
        plt.figure()
        plt.hist(lambda_, bins = list(np.linspace(0,25,50)), label=r"Sample eigenvalues", density=True)#, histtype='step')
        plt.hist(tau_quest, bins = list(np.linspace(0,25,120)), label=r"Estimated population eigenvalues", density=True)#, histtype='step')
        plt.hist(tau_pop, bins = list(np.linspace(0,25,200)), label=r"True population eigenvalues", density=True)#, histtype='step')
        plt.title("WMA sample covariance eigenvalue pdf and histogram, p="+str(p))
        plt.legend()
        plt.show()
