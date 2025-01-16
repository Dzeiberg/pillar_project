# Utilities to evaluate the model fit
import numpy as np
from scipy.stats import skewnorm


def get_cdf(u, components, weights):
    cdf = np.zeros_like(u)
    for i in range(len(components)):
        cdf += weights[i] * skewnorm.cdf(u,*components[i])
    return cdf

def get_empirical_cdf(u):
    nu = len(u)
    empirical_cdf = np.linspace(0,1,nu) + (1/nu)
    return empirical_cdf

def yang_dist(x,y,p=2):
    x = np.array(x)
    y = np.array(y)
    gt = x >= y
    dP = ((x[gt] - y[gt]).sum()**p + (y[~gt] - x[~gt]).sum()**p) ** (1/p)
    dPn = dP / sum([max(abs(xi),abs(yi),abs(xi-yi)) for xi,yi in zip(x,y)])
    return dPn

def get_cdf_dist(x, components, weights):
    """
    Calculate the distance between the empirical CDF of x and the model CDF for a given sample
    """
    x = np.array(x)
    u = sorted(np.unique(x[np.random.randint(0,len(x),size=len(x))]))
    empirical_cdf = get_empirical_cdf(u)
    model_cdf = get_cdf(u, components, weights)
    return yang_dist(empirical_cdf,model_cdf)