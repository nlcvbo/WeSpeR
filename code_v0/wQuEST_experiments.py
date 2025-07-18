import torch
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import warnings

from wQuEST import f_xi, m_idx
from wQuEST_train import wQuEST
from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from weighted_LWO_estimator import  wLWO_estimator
from lanczos import lanczos, lanczos_f, lanczos_quadrature
# import jax
# import jax.numpy as jnp
# from matfree import decomp, funm, stochtrace, eig

from wQuEST_benchmark import auto_QuEST_minimization

def wQuEST_experiment_1(tau_init = None, p = 2000, n = 20000):
    benchmark = True
    quest = True
    plots = False

    c = p/n
    
    t = torch.tensor([1.,3.,10.])
    wt = torch.tensor([0.2,0.4,0.4])
    Kt = 30
    
    mu = 0.1
    alpha = 5.#(1-1/5.5)*2
    weights = 'ewma'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    #d = torch.tensor([0.5,81/2])
    #wd = torch.tensor([1-1/80,1/80])
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_b = 100
    n_epochs_q = 100
    lr = 5e-2
    decay = 1e0
    omega = min(400, min(p,n))
    p_tau = p
    loss_type = "trap"
    verbose = True
    
    # for benchmark
    b = 1
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    
    # Sampling
    if n <= p:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n
        SLWO = wLWO_estimator(X_sample, W/n, assume_centered = True, S_r = None, est = False)
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        e_SLWO, _ = np.linalg.eigh(SLWO)
    else:
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size = (n_sample,p)) @ np.sqrt(np.diag(tau_pop))  
            S += X_sample.T @  (W[n_used:n_used+n_sample, None] * X_sample)/n
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        SLWO = shrinkage*mu_S*np.eye(p) + (1 - shrinkage)*S
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
    
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_

    if tau_init is None:
        if c >= 1:
            # tau_init = np.linspace(np.min(e_SLWO), np.max(e_SLWO), p)
            # tau_mean = tau_init.mean()
            # tau_init = tau_init/tau_mean*e_SLWO.mean()
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        else:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        tau_init = np.sort(np.random.choice(tau_init, p_tau))
        tau_init = torch.tensor(tau_init)
    
    W = torch.tensor(W)
    
    # Covariance estimation
    start_time = time.time()
    h_oracle = nl_cov_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    # h_oracle = nl_cov_shrinkage(lambda_, torch.tensor(tau_pop), torch.ones(tau_pop.shape[0])/tau_pop.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    last_time = time.time()
    print("NL shrinkage time:", last_time - start_time, "s.")
    Soracle = U @ np.diag(h_oracle) @ U.T
    cov =  np.diag(tau_pop)
    norm_cov = np.linalg.norm(cov, ord = 'fro')**2/p
    loss_S = np.linalg.norm(S - cov, ord = 'fro')**2/p
    loss_SLWO = np.linalg.norm(SLWO - cov, ord = 'fro')**2/p
    loss_Soracle = np.linalg.norm(Soracle - cov, ord = 'fro')**2/p
    PRIALS_oracle = 1-loss_Soracle/loss_S
    
    # Precision estimation
    prec = np.diag(1/tau_pop)
    t_oracle = nl_prec_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    Poracle = U @ np.diag(t_oracle) @ U.T
    norm_prec = np.linalg.norm(prec, ord = 'fro')**2/p
    loss_P = np.linalg.norm(np.linalg.pinv(S) - prec, ord = 'fro')**2/p
    loss_PLWO = np.linalg.norm(np.linalg.pinv(SLWO) - prec, ord = 'fro')**2/p
    loss_Poracle = np.linalg.norm(Poracle - prec, ord = 'fro')**2/p
    PRIALP_oracle = 1-loss_Poracle/loss_P
    
    # Optimize tau
    if quest:
        print("wQuEST:", quest) 
        start_time = time.time()
        model, tau_fit, all_tau = wQuEST(torch.tensor(lambda_), tau_init, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_q, decay = decay, loss_type = loss_type, method = method, verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        # PRIALS_train = []
        # PRIALP_train = []
        # for i in range(0,len(all_tau),5):
        #     tau_train = all_tau[i]
        #     h_lambda = nl_cov_shrinkage(lambda_, tau_train, torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        #     Snl = U @ np.diag(h_lambda) @ U.T
        #     loss_Snl = np.linalg.norm(Snl -  np.diag(tau_pop), ord = 'fro')**2/p
        #     PRIALS_train += [1-loss_Snl/loss_S]

        #     t_lambda = nl_prec_shrinkage(lambda_, tau_train, torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        #     Pnl = U @ np.diag(t_lambda) @ U.T
        #     loss_Pnl = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
        #     PRIALP_train += [1-loss_Pnl/loss_P]

        # plt.figure()
        # plt.plot(5*np.ones(len(PRIALP_train)).cumsum()-4, np.array(PRIALS_train), label="PRIAL S train")
        # plt.plot(5*np.ones(len(PRIALS_train)).cumsum()-4, PRIALS_oracle*np.ones(len(PRIALS_train)), label="PRIAL P oracle")
        # plt.title("PRIAL S train wQuEST")
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.plot(5*np.ones(len(PRIALP_train)).cumsum()-4, np.array(PRIALP_train), label="PRIAL P train")
        # plt.plot(5*np.ones(len(PRIALP_train)).cumsum()-4, PRIALP_oracle*np.ones(len(PRIALP_train)), label="PRIAL P oracle")
        # plt.title("PRIAL P train wQuEST")
        # plt.legend()
        # plt.show()

        h_lambda = nl_cov_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Snl = U @ np.diag(h_lambda) @ U.T
        loss_Snlq = np.linalg.norm(Snl -  np.diag(tau_pop), ord = 'fro')**2/p
        t_lambda = nl_prec_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Pnl = U @ np.diag(t_lambda) @ U.T
        loss_Pnlq = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
    
    # Benchmark
    if benchmark:
        print("Benchmark:", benchmark)
        start_time = time.time()
        model, tau_fit, bench_loss, all_tau = auto_QuEST_minimization(torch.tensor(S), n, p, tau_init, torch.sqrt(W), method = "Adam", n_epochs = n_epochs_b, b = b, assume_centered = False, lr = 5e-2, momentum=0., verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        # PRIALS_train = []
        # PRIALP_train = []
        # for i in range(0,len(all_tau),10):
        #     tau_train = all_tau[i]
        #     h_lambda = nl_cov_shrinkage(lambda_, tau_train, torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        #     Snl = U @ np.diag(h_lambda) @ U.T
        #     loss_Snl = np.linalg.norm(Snl - cov, ord = 'fro')**2/p
        #     PRIALS_train += [1-loss_Snl/loss_S]

        #     t_lambda = nl_prec_shrinkage(lambda_, tau_train, torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        #     Pnl = U @ np.diag(t_lambda) @ U.T
        #     loss_Pnl = np.linalg.norm(Pnl - prec, ord = 'fro')**2/p
        #     PRIALP_train += [1-loss_Pnl/loss_P]
        
        # plt.figure()
        # plt.plot(10*np.ones(len(PRIALP_train)).cumsum()-4, np.array(PRIALS_train), label="PRIAL S train")
        # plt.plot(10*np.ones(len(PRIALS_train)).cumsum()-4, PRIALS_oracle*np.ones(len(PRIALS_train)), label="PRIAL P oracle")
        # plt.title("PRIAL S train benchmark")
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.plot(10*np.ones(len(PRIALP_train)).cumsum()-4, np.array(PRIALP_train), label="PRIAL P train")
        # plt.plot(10*np.ones(len(PRIALP_train)).cumsum()-4, PRIALP_oracle*np.ones(len(PRIALP_train)), label="PRIAL P oracle")
        # plt.title("PRIAL P train benchmark")
        # plt.legend()
        # plt.show()

        h_lambda = nl_cov_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Snl = U @ np.diag(h_lambda) @ U.T
        loss_Snlb = np.linalg.norm(Snl - np.diag(tau_pop), ord = 'fro')**2/p
        t_lambda = nl_prec_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Pnl = U @ np.diag(t_lambda) @ U.T
        loss_Pnlb = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
    
    print("Norm cov:       \t", np.around(norm_cov,3))
    print("Loss/PRIAL S:   \t", np.around(loss_S,3), "\t|\t", 0)
    print("Loss/PRIAL SLWO:\t", np.around(loss_SLWO,3), "\t|\t", np.around(1-loss_SLWO/loss_S,3))
    if benchmark:
        print("Loss/PRIAL Snlb: \t", np.around(loss_Snlb,3), "\t|\t", np.around(1-loss_Snlb/loss_S,3))
    if quest:
        print("Loss/PRIAL Snlq: \t", np.around(loss_Snlq,3), "\t|\t", np.around(1-loss_Snlq/loss_S,3))
    print("Loss/PRIAL Sor: \t", np.around(loss_Soracle,3), "\t|\t", np.around(1-loss_Soracle/loss_S,3))
    # print("PRIAL Snlq/SLWO: \t", np.around(1-loss_Snlq/loss_SLWO,3))
    print()
    
    print("Norm prec:      \t", np.around(norm_prec,3))
    print("Loss/PRIAL P:   \t", np.around(loss_P,3), "\t|\t", 0)
    print("Loss/PRIAL PLWO:\t", np.around(loss_PLWO,3), "\t|\t", np.around(1-loss_PLWO/loss_P,5))
    if benchmark:
        print("Loss/PRIAL Pnlb:\t", np.around(loss_Pnlb,3), "\t|\t", np.around(1-loss_Pnlb/loss_P,3))
    if quest:
        print("Loss/PRIAL Pnlq:\t", np.around(loss_Pnlq,3), "\t|\t", np.around(1-loss_Pnlq/loss_P,3))
    print("Loss/PRIAL Por: \t", np.around(loss_Poracle,3), "\t|\t", np.around(1-loss_Poracle/loss_P,5))
    # print("PRIAL Snl/SLWO: \t", np.around(1-loss_Pnlq/loss_PLWO,3))
    
    if plots:
        # Plot the results        
        Fx = f_xi.apply
        output = Fx(tau_fit, None, d, wd, c, mu, weights, w_args, omega, method, False) 
        nu = output[-1].to(int)
        omegai = output[-nu-1:-1].to(int)
        output = output[:-nu-1]
        f = output[:output.shape[0]//3]
        F = output[output.shape[0]//3:2*output.shape[0]//3]
        xi = output[2*output.shape[0]//3:]
        plt.figure()
        plt.plot(xi.detach().numpy(), f.detach().numpy(), label="Asymptotic sample density")
        plt.hist(lambda_, bins=100, density=True, label="Sample density")
        plt.hist(tau_fit.detach().numpy(), weights=3*np.ones(tau_fit.shape[0])/tau_fit.shape[0]*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=100, label="Asymptotic population density")
        plt.hist(t.detach().numpy(), weights=4*wt.detach().numpy()*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=150, label="Population density")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()
        
        output_i = Fx(tau_init, None, d, wd, c, mu, weights, w_args, omega, method, False) 
        nu = output_i[-1].to(int)
        omegai = output_i[-nu-1:-1].to(int)
        output_i = output_i[:-nu-1]
        f_i = output_i[:output_i.shape[0]//3]
        F_i = output_i[output_i.shape[0]//3:2*output_i.shape[0]//3]
        xi_i = output_i[2*output_i.shape[0]//3:]
        
        plt.figure()
        plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Fit sample cdf")
        plt.plot(xi_i.detach().numpy(), F_i.detach().numpy(), label="Init sample cdf")
        plt.hist(lambda_, bins=200, cumulative = True, histtype="step", density=True, label="Sample cdf")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.hist(tau_fit.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Fit population cdf")
        plt.hist(tau_init.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Init population cdf")
        plt.hist(t.detach().numpy(), weights=wt.detach().numpy(), cumulative = True, histtype="step", bins=200, label="Population density")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()    
    
        return model, tau_fit.detach(), tau_init, f, F, xi, f_i, F_i, xi_i, lambda_
    return model, tau_fit.detach(), tau_init, None, None, None, None, None, None, lambda_

def wQuEST_experiment_2(tau_init = None, p = 2000, n = 20000):
    dtype = np.float64
    c = p/n
    
    t = torch.tensor([1.,3.,10.])
    wt = torch.tensor([0.2,0.4,0.4])
    Kt = 30
    
    mu = 0.1
    alpha = 0.1#(1-1/5.5)*2
    weights = 'ewma'   
    method = 'root'
    Kd = 1

    num_probes = 1
    K = min(p,1000)
    
    Nd = 200
    #d = torch.tensor([0.5,81/2])
    #wd = torch.tensor([1-1/80,1/80])
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_q = 100
    lr = 5e-2
    decay = 1e0
    omega = min(400, min(p,n))
    p_tau = K
    loss_type = "trap"
    verbose = True

    if p < 10000:
        ortho = True
    else:
        ortho = True

    # Draw v for regression
    v = np.random.normal(size=p).astype(dtype)
    v = v / np.linalg.norm(v)
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5), dtype=dtype)
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    W = W.astype(dtype)
    
    # Sampling
    if n <= p:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n

        mu_S = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = ((X_sample**2).sum(axis=1)*2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        SLWO = shrinkage*mu*np.eye(p) + (1 - shrinkage)*S
        
        # Init Lanczos 
        _, _, _ = lanczos_quadrature(S[:10, :10], v[:10], int(3), ortho = ortho)
        start_time = time.time()
        xi_S, F_S = lanczos_quadrature(S, num_probes, K, ortho = ortho)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        p2 = K
        n2 = int(p2/c)
        lambda_ = np.zeros(p2)
        for kappa in range(max(p2-n2,0), p2):
            j_k = max(np.argmax((((kappa+1/2) - p2*F_S) < 0).astype(int)) - 1, 0)
            j_kp = max(np.argmax((((kappa+1/2) - p2*F_S) < 0).astype(int)), 0)
            if F_S[j_kp] - F_S[j_k] > 0:
                lambda_[kappa] = (((kappa+1/2)/p2 - F_S[j_k])*xi_S[j_k] + (F_S[j_kp] - (kappa+1/2)/p2)*xi_S[j_kp])/(F_S[j_kp] - F_S[j_k])
            else:
                lambda_[kappa] = (xi_S[j_k]+xi_S[j_kp])/2

        e_SLWO = shrinkage*mu + (1 - shrinkage)*lambda_
    else:
        start_time = time.time()
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p), dtype=dtype)
        while n_left > 0:
            n_sample = min(n_left, p//4)
            rng = np.random.default_rng()
            X_sample = rng.standard_normal(size = (n_sample,p), dtype=dtype) * np.sqrt(tau_pop)[None,:]
            # X_sample = np.random.normal(size = (n_sample,p), dtype=dtype)
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            X_sample = np.sqrt(W[n_used:n_used+n_sample, None]/n) * X_sample
            X_sample = X_sample.T @  X_sample
            S += X_sample

            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S)/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        delta2 = S2 - mu_S**2
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        last_time = time.time()
        print("Sampling time:", last_time - start_time, "s.")

        start_time = time.time()
        xi_S, F_S = lanczos_quadrature(S, num_probes, K, ortho = ortho)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        p2 = K
        n2 = int(p2/c)
        lambda_ = np.zeros(p2)
        for kappa in range(max(p2-n2,0), p2):
            j_k = max(np.argmax((((kappa+1/2) - p2*F_S) < 0).astype(int)) - 1, 0)
            j_kp = max(np.argmax((((kappa+1/2) - p2*F_S) < 0).astype(int)), 0)
            if F_S[j_kp] - F_S[j_k] > 0:
                lambda_[kappa] = (((kappa+1/2)/p2 - F_S[j_k])*xi_S[j_k] + (F_S[j_kp] - (kappa+1/2)/p2)*xi_S[j_kp])/(F_S[j_kp] - F_S[j_k])
            else:
                lambda_[kappa] = (xi_S[j_k]+xi_S[j_kp])/2
    
        e_SLWO = shrinkage*mu + (1 - shrinkage)*lambda_

        plt.figure()
        plt.plot(xi_S, F_S)
        plt.show()

    if tau_init is None:
        if c >= 1:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        else:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        tau_init = np.sort(np.random.choice(tau_init, p_tau))
        tau_init = torch.tensor(tau_init)
    
    W = torch.tensor(W)
    
    # Optimize tau
    print("wQuEST:") 
    start_time = time.time()
    model, tau_fit, all_tau = wQuEST(torch.tensor(lambda_), tau_init, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_q, decay = decay, loss_type = loss_type, method = method, verbose = verbose)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")

    start_time = time.time()
    f = lambda x: nl_prec_shrinkage(x, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    wnl = lanczos_f(S, v, K, f, ortho = ortho)
    last_time = time.time()
    print("wnl compute time:", last_time - start_time, "s.")

    start_time = time.time()
    f = lambda x: nl_prec_shrinkage(x, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    wor = lanczos_f(S, v, K, f, ortho = ortho)
    last_time = time.time()
    print("wor compute time:", last_time - start_time, "s.")
    
    start_time = time.time()
    wna = np.linalg.solve(S, v)
    last_time = time.time()
    print("wna compute time:", last_time - start_time, "s.")
    
    # SLWO = shrinkage*mu*np.eye(p) + (1 - shrinkage)*S
    S *= 1 - shrinkage
    S[np.diag_indices(p)] += shrinkage*mu_S
    start_time = time.time()
    wls = np.linalg.solve(S, v)
    last_time = time.time()
    print("wls compute time:", last_time - start_time, "s.")

    wpop = v/tau_pop
    
    print("norm:", np.linalg.norm(wpop)**2/p)
    print("wnl: \t", np.around(np.linalg.norm(wnl-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wnl-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wor: \t", np.around(np.linalg.norm(wor-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wor-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wls: \t", np.around(np.linalg.norm(wls-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wls-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wna: \t", np.around(np.linalg.norm(wna-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wna-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))

    return lambda_, wnl, wor, wpop, wna, wls


def wQuEST_experiment_3(H_distrib = 1, tau_init = None, p = 2000, n = 20000):
    benchmark = True
    quest = True
    plots = True

    c = p/n
    
    x = np.linspace(0,1,2000)
    H1 = 1 - (1-x**3)**(1/3)
    H2 = (1 - (1-x)**3)**(1/3)
    H3 = (1 - np.abs(1-2*x)**3)**(1/3)/2
    H3[x>1/2] = 1 - H3[x>1/2]
    H4 = 1/2 - (1 - (2*x)**3)**(1/3)/2
    H4[x>1/2] = 1/2 + (1 - (2-2*x[x>1/2])**3)**(1/3)/2
    if H_distrib == 1:
        H = H1
    elif H_distrib == 2:
        H = H2
    elif H_distrib == 3:
        H = H3
    else:
        H = H4
    t = torch.zeros(p)
    for kappa in range(p):
        j_k = max(np.argmax((((kappa+1/2) - p*H) < 0).astype(int)) - 1, 0)
        j_kp = max(np.argmax((((kappa+1/2) - p*H) < 0).astype(int)), 0)
        if H[j_kp] - H[j_k] > 0:
            t[kappa] = (((kappa+1/2)/p - H[j_k])*x[j_k] + (H[j_kp] - (kappa+1/2)/p)*x[j_kp])/(H[j_kp] - H[j_k])
        else:
            t[kappa] = (x[j_k]+x[j_kp])/2
    t = torch.tensor(1 + 9*t)
    wt = torch.ones(p)/p
    Kt = 30
    
    mu = 0.1
    alpha = 1.#(1-1/5.5)*2
    weights = 'unif'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    #d = torch.tensor([0.5,81/2])
    #wd = torch.tensor([1-1/80,1/80])
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_b = 500
    n_epochs_q = 500
    lr = 5e-2
    decay = 1e0
    omega = min(400, min(p,n))
    p_tau = p
    loss_type = "trap"
    verbose = True
    
    # for benchmark
    b = 1
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    tau_pop = t.numpy()
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    
    # Sampling
    if n <= p:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n
        SLWO = wLWO_estimator(X_sample, W/n, assume_centered = True, S_r = None, est = False)
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        e_SLWO, _ = np.linalg.eigh(SLWO)
    else:
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size = (n_sample,p)) @ np.sqrt(np.diag(tau_pop))  
            S += X_sample.T @  (W[n_used:n_used+n_sample, None] * X_sample)/n
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        SLWO = shrinkage*mu*np.eye(p) + (1 - shrinkage)*S
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
    
        e_SLWO = shrinkage*mu + (1 - shrinkage)*lambda_

    if tau_init is None:
        if c >= 1:
            # tau_init = np.linspace(np.min(e_SLWO), np.max(e_SLWO), p)
            # tau_mean = tau_init.mean()
            # tau_init = tau_init/tau_mean*e_SLWO.mean()
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        else:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        tau_init = np.sort(np.random.choice(tau_init, p_tau))
        tau_init = torch.tensor(tau_init)
    
    W = torch.tensor(W)
    
    # Covariance estimation
    start_time = time.time()
    h_oracle = nl_cov_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    # h_oracle = nl_cov_shrinkage(lambda_, torch.tensor(tau_pop), torch.ones(tau_pop.shape[0])/tau_pop.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    last_time = time.time()
    print("NL shrinkage time:", last_time - start_time, "s.")
    Soracle = U @ np.diag(h_oracle) @ U.T
    cov =  np.diag(tau_pop)
    norm_cov = np.linalg.norm(cov, ord = 'fro')**2/p
    loss_S = np.linalg.norm(S - cov, ord = 'fro')**2/p
    loss_SLWO = np.linalg.norm(SLWO - cov, ord = 'fro')**2/p
    loss_Soracle = np.linalg.norm(Soracle - cov, ord = 'fro')**2/p
    PRIALS_oracle = 1-loss_Soracle/loss_S
    
    # Precision estimation
    prec = np.diag(1/tau_pop)
    t_oracle = nl_prec_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    Poracle = U @ np.diag(t_oracle) @ U.T
    norm_prec = np.linalg.norm(prec, ord = 'fro')**2/p
    loss_P = np.linalg.norm(np.linalg.pinv(S) - prec, ord = 'fro')**2/p
    loss_PLWO = np.linalg.norm(np.linalg.pinv(SLWO) - prec, ord = 'fro')**2/p
    loss_Poracle = np.linalg.norm(Poracle - prec, ord = 'fro')**2/p
    PRIALP_oracle = 1-loss_Poracle/loss_P
    
    # Optimize tau
    if quest:
        print("wQuEST:", quest) 
        start_time = time.time()
        model, tau_fit, all_tau = wQuEST(torch.tensor(lambda_), tau_init, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_q, decay = decay, loss_type = loss_type, method = method, verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        h_lambda = nl_cov_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Snl = U @ np.diag(h_lambda) @ U.T
        loss_Snlq = np.linalg.norm(Snl -  np.diag(tau_pop), ord = 'fro')**2/p
        t_lambda = nl_prec_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Pnl = U @ np.diag(t_lambda) @ U.T
        loss_Pnlq = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
    
        # Q-Q plots:
        plt.figure()
        plt.plot(tau_pop, np.sort(lambda_), label = r'Empirical $F$')
        plt.plot(tau_pop, np.sort(tau_fit.detach().numpy()), label = '(MD) algorithm')


    # Benchmark
    if benchmark:
        print("Benchmark:", benchmark)
        start_time = time.time()
        model, tau_fit, bench_loss, all_tau = auto_QuEST_minimization(torch.tensor(S), n, p, tau_init, torch.sqrt(W), method = "Adam", n_epochs = n_epochs_b, b = b, assume_centered = False, lr = 5e-2, momentum=0., verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        h_lambda = nl_cov_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Snl = U @ np.diag(h_lambda) @ U.T
        loss_Snlb = np.linalg.norm(Snl - np.diag(tau_pop), ord = 'fro')**2/p
        t_lambda = nl_prec_shrinkage(lambda_, tau_fit.detach(), torch.ones(tau_fit.shape[0])/tau_fit.shape[0], d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
        Pnl = U @ np.diag(t_lambda) @ U.T
        loss_Pnlb = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
        
        # Q-Q plots:
        plt.plot(tau_pop, np.sort(tau_fit.detach().numpy()), label = '(SD) algorithm')
        plt.plot(tau_pop, tau_pop, label = 'True population spectrum')
        plt.xlabel(r"$H$")
        plt.ylabel(r"$\hat H$")
        if H_distrib == 1:
            plt.title(r"Q-Q plot for $H_1$")
        elif H_distrib == 2:
            plt.title(r"Q-Q plot for $H_2$")
        elif H_distrib == 3:
            plt.title(r"Q-Q plot for $H_3$")
        else:
            plt.title(r"Q-Q plot for $H_4$")
        plt.legend()
        plt.show()
    print("Norm cov:       \t", np.around(norm_cov,3))
    print("Loss/PRIAL S:   \t", np.around(loss_S,3), "\t|\t", 0)
    print("Loss/PRIAL SLWO:\t", np.around(loss_SLWO,3), "\t|\t", np.around(1-loss_SLWO/loss_S,3))
    if benchmark:
        print("Loss/PRIAL Snlb: \t", np.around(loss_Snlb,3), "\t|\t", np.around(1-loss_Snlb/loss_S,3))
    if quest:
        print("Loss/PRIAL Snlq: \t", np.around(loss_Snlq,3), "\t|\t", np.around(1-loss_Snlq/loss_S,3))
    print("Loss/PRIAL Sor: \t", np.around(loss_Soracle,3), "\t|\t", np.around(1-loss_Soracle/loss_S,3))
    # print("PRIAL Snlq/SLWO: \t", np.around(1-loss_Snlq/loss_SLWO,3))
    print()
    
    print("Norm prec:      \t", np.around(norm_prec,3))
    print("Loss/PRIAL P:   \t", np.around(loss_P,3), "\t|\t", 0)
    print("Loss/PRIAL PLWO:\t", np.around(loss_PLWO,3), "\t|\t", np.around(1-loss_PLWO/loss_P,5))
    if benchmark:
        print("Loss/PRIAL Pnlb:\t", np.around(loss_Pnlb,3), "\t|\t", np.around(1-loss_Pnlb/loss_P,3))
    if quest:
        print("Loss/PRIAL Pnlq:\t", np.around(loss_Pnlq,3), "\t|\t", np.around(1-loss_Pnlq/loss_P,3))
    print("Loss/PRIAL Por: \t", np.around(loss_Poracle,3), "\t|\t", np.around(1-loss_Poracle/loss_P,5))
    # print("PRIAL Snl/SLWO: \t", np.around(1-loss_Pnlq/loss_PLWO,3))

    if plots:
        # Plot the results        
        Fx = f_xi.apply
        output = Fx(tau_fit, None, d, wd, c, mu, weights, w_args, omega, method, False) 
        nu = output[-1].to(int)
        omegai = output[-nu-1:-1].to(int)
        output = output[:-nu-1]
        f = output[:output.shape[0]//3]
        F = output[output.shape[0]//3:2*output.shape[0]//3]
        xi = output[2*output.shape[0]//3:]
        plt.figure()
        plt.plot(xi.detach().numpy(), f.detach().numpy(), label="Asymptotic sample density")
        plt.hist(lambda_, bins=100, density=True, label="Sample density")
        plt.hist(tau_fit.detach().numpy(), weights=3*np.ones(tau_fit.shape[0])/tau_fit.shape[0]*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=100, label="Asymptotic population density")
        plt.hist(t.detach().numpy(), weights=4*wt.detach().numpy()*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=150, label="Population density")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()
        
        output_i = Fx(tau_init, None, d, wd, c, mu, weights, w_args, omega, method, False) 
        nu = output_i[-1].to(int)
        omegai = output_i[-nu-1:-1].to(int)
        output_i = output_i[:-nu-1]
        f_i = output_i[:output_i.shape[0]//3]
        F_i = output_i[output_i.shape[0]//3:2*output_i.shape[0]//3]
        xi_i = output_i[2*output_i.shape[0]//3:]
        
        plt.figure()
        plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Fit sample cdf")
        plt.plot(xi_i.detach().numpy(), F_i.detach().numpy(), label="Init sample cdf")
        plt.hist(lambda_, bins=200, cumulative = True, histtype="step", density=True, label="Sample cdf")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.hist(tau_fit.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Fit population cdf")
        plt.hist(tau_init.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Init population cdf")
        plt.hist(t.detach().numpy(), weights=wt.detach().numpy(), cumulative = True, histtype="step", bins=200, label="Population density")
        plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
        plt.legend()
        plt.show()    
    
    return model, tau_fit.detach(), tau_init, None, None, None, None, None, None, lambda_

if __name__ == '__main__':
    try:
        matplotlib.use('tkagg')
        matplotlib.pyplot.ion()
    except:
        pass
    
    warnings.filterwarnings('ignore')
    
    tau_fit = None
    for i in range(10):
        print("Experiment", i)
        for p in [50]:#, 100, 200, 500, 1000, 2000]:
            print("p =", p)
            _ = wQuEST_experiment_1(tau_init = None, p = p, n = 10*p)
    
    # tau_fit = None
    # model, tau_fit, tau_init, f, F, xi, f_i, F_i, xi_i, lambda_ = wQuEST_experiment_1(tau_init = None, p = 10000, n = 100000)

    # tau_fit = None
    # for p in [50000]:
    #     print("p =", p)
    #     lambda_, _, _, _, _, _ = wQuEST_experiment_2(tau_init = None, p = p, n = 2*p)

        
    # tau_fit = None
    # model, tau_fit, tau_init, f, F, xi, f_i, F_i, xi_i, lambda_ = wQuEST_experiment_3(H_distrib = 1, tau_init = tau_fit, p = 1000, n = 3000)
    
    # tau_fit = None
    # model, tau_fit, tau_init, f, F, xi, f_i, F_i, xi_i, lambda_ = wQuEST_experiment_3(H_distrib = 2, tau_init = tau_fit, p = 1000, n = 3000)
    
    # tau_fit = None
    # model, tau_fit, tau_init, f, F, xi, f_i, F_i, xi_i, lambda_ = wQuEST_experiment_3(H_distrib = 3, tau_init = tau_fit, p = 1000, n = 3000)
    
    # tau_fit = None
    # model, tau_fit, tau_init, f, F, xi, f_i, F_i, xi_i, lambda_ = wQuEST_experiment_3(H_distrib = 4, tau_init = tau_fit, p = 1000, n = 3000)

    # exp = False
    # if exp:
        # import numpy as np
        # import scipy as sp
        # for p in [35000, 40000, 30000, 25000]:
        #     A = np.random.normal(size=(p,p)).astype(np.float32)
        #     A = A + A.T
        #     print("p =", p)
        #     start_time = time.time()
    
        #     _, _ = sp.linalg.eigh(A)
        #     last_time = time.time()
        #     print("diag compute time:", last_time - start_time, "s.")
