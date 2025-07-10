import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from numpy.polynomial import polynomial as poly
from WeSpeR_support_identification_unif import find_support_unif
from WeSpeR_support_identification_ewma import find_support_exp
from WeSpeR_support_identification_Ndiracs import find_support_Ndiracs

def zp(u, w, t, alpha, c):
    beta = alpha/(1-np.exp(-alpha))
    g = c*(w[None,:]*t[None,:]/(t[None,:]-u[:,None])).sum(axis=1)
    ugp = c*(w[None,:]*t[None,:]/(t[None,:]-u[:,None])**2).sum(axis=1)*u
    rst = - (g + ugp)*beta*np.exp(-alpha) - alpha/(1-np.exp(alpha*g))**2*((g+ugp)*(1-np.exp(alpha*g)) + ugp*g*alpha*np.exp(alpha*g))
    
    g1 = (w[None,:]*t[None,:]/(t[None,:]-u[:,None])).sum(axis=1)
    g2  = (w[None,:]*t[None,:]/(t[None,:]-u[:,None])**2).sum(axis=1)
    g3 = (w[None,:]*t[None,:]**2/(t[None,:]-u[:,None])**2).sum(axis=1)
    rst = -c*g3*(beta*np.exp(-alpha) + alpha/(1-np.exp(alpha*c*g1))) - c*u*g1*alpha**2*g2*np.exp(alpha*c*g1)/(1-np.exp(alpha*c*g1))**2
    return rst

def z_exp(u, w, t, alpha, c, verbose = False):
    beta = alpha/(1-np.exp(-alpha))
    g1 = c*(w[None,:]*t[None,:]/(t[None,:]-u[:,None])).sum(axis=1)
    m = beta*np.exp(-alpha) + alpha/(1-np.exp(alpha*g1))
    rst = -u*g1*m
    return rst

def mld_unif(x, alpha):
    return 1 + x/alpha*np.log(1 + alpha/(1-alpha/2-x))

def mld_unifp(x,alpha):
    return np.log(1 + alpha/(1-alpha/2-x))/alpha + x/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x))

def mld_unifpp(x,alpha):
    return 2/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x)) - 2*x*(x-1)/((1-alpha/2-x)**2 + alpha*(1-alpha/2-x))**2

def mld_exp(x, alpha):
    beta = alpha/(1-np.exp(-alpha))
    return 1/alpha*np.log(1 + alpha/(beta*np.exp(-alpha)-x))

def mld_expp(x,alpha):
    beta = alpha/(1-np.exp(-alpha))
    return 1/((beta*np.exp(-alpha)-x)**2 + alpha*(beta*np.exp(-alpha)-x))

def mld_exppp(x,alpha):
    beta = alpha/(1-np.exp(-alpha))
    return +(2*(beta*np.exp(-alpha)-x)+alpha)/((beta*np.exp(-alpha)-x)**2 + alpha*(beta*np.exp(-alpha)-x))**2

def z_unif(u, w, t, alpha, c, verbose = False):
    g1 = c*(w[None,:]*t[None,:]/(t[None,:]-u[:,None])).sum(axis=1)
    alpha_d = alpha/2
    m_init = 1 - 1/2/g1 -np.sqrt(1+4*g1*(g1-1)*alpha_d**2)/2/g1 # mldm for 2 diracs with alpha_d = alpha/2
    m = m_init
    for i in range(m_init.shape[0]):
        f = lambda x: mld_unif(x, alpha)-g1[i]
        fp = lambda x: mld_unifp(x, alpha)
        m[i] = root_scalar(f, x0=m[i], fprime=fp, method="newton").root
    rst = -u*g1*m
    return rst

def mld_Ndiracs(x, wd, d):
    if len(x.shape) == 1:
        return (wd[None,:]*d[None,:]/(d[None,:] - x[:, None])).sum(axis=1)
    else:
        return (wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])).sum(axis=2)

def mld_Ndiracsp(x, wd, d):
    if len(x.shape) == 1:
        return (wd[None,:]*d[None,:]/(d[None,:] - x[:, None])**2).sum(axis=1)
    else:
        return (wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])**2).sum(axis=2)

def mld_Ndiracspp(x, wd, d):
    if len(x.shape) == 1:
        return 2*(wd[None,:]*d[None,:]/(d[None,:] - x[:, None])**3).sum(axis=1)
    else:
        return 2*(wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])**3).sum(axis=2)

def z_Ndiracs_poly(u, w, t, d, wd, c, verbose = False):
    M = d.shape[0]
    g1 = c*(w[None,:]*t[None,:]/(t[None,:]-u[:,None])).sum(axis=1)
    
    P = poly.Polynomial((0))
    for i in range(M):
        R = poly.Polynomial((wd[i]*d[i]))
        for j in range(M):
            if j != i:
                R = poly.polymul(R, poly.Polynomial((d[j], -1)))
        P = poly.polyadd(P, R)[0]
    
    Q = poly.Polynomial((1))
    for i in range(M):
        Q = poly.polymul(Q, poly.Polynomial((d[i], -1)))[0]
    
    m = np.zeros((u.shape[0],M))
    for i in range(len(u)):
        R = poly.polyadd(P, poly.polymul(poly.Polynomial((-g1[i])),Q))[0]
        try:
            #roots =  np.sort(find_roots(R, R.deriv(), eps=1e-12, Mf=10, verbose=False)[0].real)
            roots =  np.sort(R.roots().real)
            if roots[0] < 0:
                roots = np.concatenate([roots[1:],roots[:1]])
            m[i] = roots
            
        except:
            m[i] = np.inf
    rst = -u[:,None]*g1[:,None]*m
    
    #see p.127
    mlhp = (w[None,:]*t[None,:]/(t[None,:]-u[:,None])**2).sum(axis=1)[:,None]
    mld = mld_Ndiracs(m, wd, d)
    mldp = mld_Ndiracsp(m, wd, d)
    rstp = -mld*m - u[:,None]*(mld/mldp + m)*c*mlhp
    
    return rst.T, rstp.T

def plot_xF(n, p, wd, d, alpha, tau_pop ,t ,w ,u ,weight="Ndiracs", eps=1e-1, verbose=False):
    # Sampling
    c = p/n
    
    if weight == "exp":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weight == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weight == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weight == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    Wsq = np.sqrt(W)[:,None]
    
    X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))
    ewma_cov = (X_sample*Wsq).T @ (X_sample*Wsq)/X_sample.shape[0]
    lambda_ = np.linalg.eigh(ewma_cov)[0]
    
       
    plt.figure()
    
    if weight == "exp":
        zu = z_exp(u, w, t, alpha, c)  
        sup = find_support_exp(c, alpha, t, w, eps=eps, verbose=verbose)
        
        plt.plot(u, zu, label=r"$x_F(-1/u)$")
        for x in sup:
            plt.plot(np.linspace(-1,np.max(u),2), x*np.ones(2))
            
    elif weight == "unif":
        zu = z_unif(u, w, t, alpha, c, verbose = verbose)
        sup = find_support_unif(c, alpha, t, w, eps=eps, verbose=verbose)
        
        plt.plot(u, zu, label=r"$x_F(-1/u)$")
        for x in sup:
            plt.plot(np.linspace(-1,np.max(u),2), x*np.ones(2))
            
    elif weight == "Ndiracs":
        d_emp = d
        wd_emp = np.floor(wd*n)
        wd_emp[0] = n - wd_emp[1:].sum()
        wd_emp = wd_emp/wd_emp.sum()
        
        z, zp = z_Ndiracs_poly(u, w, t, d_emp, wd_emp, c, verbose = verbose)
        sup = find_support_Ndiracs(c, wd_emp, d_emp, t, w, eps=eps, verbose=verbose)      
                
        for k in range(1, d.shape[0]+1):
            label = r"$x_F^{("+str(k)+")}(-1/u)$"
            plt.plot(u, z[k-1], label=label)
        for x in sup:
            plt.plot(np.linspace(-1,np.max(u),2), x*np.ones(2))
            
    plt.plot(np.zeros(lambda_.shape), lambda_, linestyle='', marker='.', label="Empirical eigenvalues")
    plt.xlabel("u")
    plt.ylabel("x")
    plt.title("Spectrum support identification, c="+str(p/n))
    plt.legend()
    plt.ylim((0, lambda_.max()*1.2))
    plt.show()

def experiment_1():
    n = 10000
    p = 1000
    alpha = 1e0
    weight = "unif"
    
    u = np.linspace(0,20,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_2():
    n = 10000
    p = 1000
    weight = "Ndiracs"
    d = np.array([1/2,1/2*(1-1/80)*80])
    wd = np.array([1 - 1/80,1/80])
    
    u = np.linspace(0,35,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, wd, d, None, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_3():
    n = 10000
    p = 1000
        
    alpha = 5e0
    weight = "exp"
    
    u = np.linspace(0,20,5010)  
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_4():
    n = 10000
    p = 1000
    alpha = 1e1
    weight = "exp"
    
    u = np.linspace(0,20,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_5():
    n = 10000
    p = 1000
    alpha = 1e0
    weight = "unif"
    
    u = np.linspace(0,20,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_6():
    n = 10000
    p = 5000
    alpha = 1e0
    weight = "unif"
    
    u = np.linspace(0,20,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_7():
    n = 40000
    p = 4000
    weight = "Ndiracs"
    d = np.array([1/2,1/2*(1-1/80)*80])
    wd = np.array([1 - 1/80,1/80])
    
    u = np.linspace(0,30,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, wd, d, None, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_8():
    n = 40000
    p = 4000
    weight = "Ndiracs"
    d = np.array([0.5, 1, 4, 10, 50])
    wd = np.array([2, 1, 1/4, 1/10, 1/50])
    wd /= wd.sum()
    d /= (wd*d).sum()
    
    u = np.linspace(0,30,5010)    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, wd, d, None, tau_pop, t, w , u, weight=weight, verbose=False)
    
def experiment_9():
    n = 1000
    p = 2000
        
    alpha = 5e0
    weight = "exp"
    
    u = np.linspace(-10,50,5010)  
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    t = np.array([1,3,10])
    w = np.array([0.2, 0.4, 0.4])
    
    plot_xF(n, p, None, None, alpha, tau_pop, t, w , u, weight=weight, verbose=False)
    
if __name__ == '__main__':
    # Execute this will reproduce every figure of support identification of the corpus and the appendix, this can take a while /!\
    print("Executing support identification experiments.")
    # experiment_1()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # experiment_6()
    # experiment_7()
    # experiment_8()
    experiment_9()
