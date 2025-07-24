import numpy as np
import scipy as sp
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from WeSpeR_MD_utils import WeSpeR_MD_model, WeSpeR_MD_minimization
from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from weighted_LWO_estimator import wLWO_estimator
from lanczos import lanczos, lanczos_f, lanczos_quadrature, lanczos_fA

class WeSpeR_HD(BaseEstimator, TransformerMixin):
    """
    A Scikit-Learn-compatible covariance estimator template.
    
    Parameters
    ----------
    bias : bool, default=False
        If True, normalize by N. If False, normalize by N-1.
    assume_centered : bool, default=False
        If True, data will not be centered before computing covariance.
    """

    def __init__(self, bias=True, assume_centered=False):
        self.bias = bias
        self.assume_centered = assume_centered

    def fit_from_S(self, S, W, y=None, shrinkage = 0., num_probes = 1, K = 1000, ortho = True, p_tau = None, c = None, tau_init = None, method = "root", n_epochs = 100, wt = None, d = torch.ones(1), wd = torch.ones(1), mu = 0.1, weights = 'Ndiracs', w_args = None, omega = 100, loss_type = 'trap', assume_centered = False, lr = 5e-2, momentum=0., verbose = True):
        """
        Estimate the covariance matrix from sample covariance.

        Parameters
        ----------
        S : array-like of shape (n_features, n_features)
            Training sample covariance.
        
        W : array-like of shape (n_samples)
            Training weight vector.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        p = S.shape[0]
        n = W.shape[0]
        self.n, self.p = n, p

        if p_tau is None:
            p_tau = p
        self.p_tau = p_tau 

        ddof = 0 if self.bias else (W**2).sum()/n**2
        if c is None:
            self.c = p/(n-ddof)
        else:
            self.c = c
        
        xi_S, F_S = lanczos_quadrature(S, num_probes, K, ortho = ortho)
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
        mu_S = np.trace(S)/p
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_

        if tau_init is None:
            if self.c >= 1:
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

        model, tau_fit, loss, all_tau = WeSpeR_MD_minimization(torch.tensor(lambda_), tau_init, wt = wt, d = d, wd = wd, c = self.c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs, loss_type = loss_type, method = method, verbose = verbose)

        self.model_ = model
        self.tau_fit_ = tau_fit
        self.loss = loss
        self.lambda_ = lambda_
        self.S = S
        return self

    def fit(self, X, W, y=None, c = None, p_tau = None, num_probes = 1, K = 1000, ortho = True, tau_init = None, method = "root", n_epochs = 100, wt = None, d = torch.ones(1), wd = torch.ones(1), mu = 0.1, weights = 'Ndiracs', w_args = None, omega = 100, loss_type = 'trap', assume_centered = False, lr = 5e-2, momentum=0., verbose = True):
        """
        Estimate the covariance matrix from data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        W : array-like of shape (n_samples)
            Training weight vector.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X, dtype=np.float64)
        n, p = X.shape
        self.n, self.p = n, p

        if p_tau is None:
            p_tau = p
        self.p_tau = p_tau 

        if not self.assume_centered:
            self.location_ = X.mean(axis=0)
            X = X - self.location_[None,:]
        else:
            self.location_ = np.zeros(X.shape[1], dtype=X.dtype)

        ddof = 0 if self.bias else (W**2).sum()/n**2
        if c is None:
            self.c = p/(n-ddof)
        else:
            self.c = c
        
        if n <= p and p <= 5000:
            S = X.T @  (W[:, None] * X)/(n-ddof)
            SLWO = wLWO_estimator(X, W/n, assume_centered = assume_centered, S_r = None, est = False)
            lambda_, U = sp.linalg.eigh(S)
            e_SLWO = np.linalg.eigvalsh(SLWO)
        else:
            n_left = n
            n_used = 0
            X2sum = np.zeros(n)
            S = np.zeros((p,p))
            while n_left > 0:
                n_sample = min(n_left, p)
                S += X.T @  (W[n_used:n_used+n_sample, None] * X)/n
                X2sum[n_used:n_used+n_sample] = (X**2).sum(axis=1)
                n_used += n_sample
                n_left -= n_sample

            mu_S = np.trace(S)/p
            delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
            S2 = np.linalg.norm(S, ord = 'fro')**2/p
            beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
            beta2 = beta2/(1-(W**2).sum()/n**2)
            beta2 = min(beta2, delta2)
            
            shrinkage = beta2/delta2            
        
        return self.fit_from_S(S, W, y=None, shrinkage = shrinkage, num_probes = num_probes, K = 1000, ortho = ortho, p_tau = p_tau, c = c, tau_init = tau_init, method = method, n_epochs = n_epochs, wt = wt, d =d, wd = wd, mu = mu, weights = weights, w_args = w_args, omega = omega, loss_type = loss_type, assume_centered = assume_centered, lr = lr, momentum = momentum, verbose = verbose)
        

    def transform(self, X):
        """
        Optionally center the data using the location_.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_centered : ndarray
        """
        check_is_fitted(self, "location_")
        X = self._validate_data(X, reset=False)

        if not self.assume_centered:
            return X - self.location_
        return X
    
    def get_covariance(self, num_probes = 1, K = 1000, ortho = True, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated covariance matrix, not precise. Use get_covariance_operator instead. """
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "covariance_")
        except NotFittedError:  
            f = lambda x: nl_cov_shrinkage(x, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = verbose).numpy()
            self.covariance_ = lanczos_fA(self.S, K, f, ortho = ortho)
        return self.covariance_
    
    def get_precision(self, num_probes = 1, K = 1000, ortho = True, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated precision matrix, not precise. Use get_precision_operator instead. """
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "precision_")
        except NotFittedError:  
            f = lambda x: nl_prec_shrinkage(x, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = verbose).numpy()
            self.precision_ = lanczos_fA(self.S, K, f, ortho = ortho)
        return self.precision_

    def get_covariance_operator(self, K = 1000, ortho = True, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated covariance matrix operator."""
        check_is_fitted(self, "model_")
        f = lambda x: nl_cov_shrinkage(x, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = verbose).numpy()
        return lambda v: lanczos_f(self.S, v, K, f, ortho = ortho)
    
    def get_precision_operator(self, K = 1000, ortho = True, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated precision matrix operator."""
        check_is_fitted(self, "model_")
        f = lambda x: nl_prec_shrinkage(x, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = verbose).numpy()
        return lambda v: lanczos_f(self.S, v, K, f, ortho = ortho)
    
    def get_tau(self):
        """Return the estimated population eigenvalues."""
        check_is_fitted(self, "tau_fit_")
        return self.tau_fit_

    def error_norm(self, comp_cov, norm='frobenius'):
        """
        Compute the error norm between estimated covariance and a given one.

        Parameters
        ----------
        comp_cov : ndarray of shape (n_features, n_features)
            The covariance to compare against.

        norm : {'frobenius', 'spectral'}, default='frobenius'
            The norm used to compute the error.

        Returns
        -------
        norm_value : float
        """
        est_cov = self.get_covariance()
        diff = est_cov - comp_cov

        if norm == 'frobenius':
            return np.sqrt(np.sum(diff ** 2))
        elif norm == 'spectral':
            return np.linalg.norm(diff, ord=2)
        else:
            raise ValueError("Unsupported norm type.")
