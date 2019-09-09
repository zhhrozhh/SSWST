## use GEN to calculate emperical p-value table


import numpy as np
import scipy.stats as csc
import pandas as pd


def ES(aa = 30,bb = 4758):
    """
        aa(int), 1/dp
        bb(int), (u-s)/dt
        return(float), the W-value
    """
    invdp = aa
    invdt = bb

    dp = 1/invdp
    dt = 1/invdt
    RE = []

    b1s = csc.uniform(loc = 0.01, scale = 1.5).rvs(size = invdt)
    b2s = csc.uniform(loc = 0.01, scale = 1.5).rvs(size = invdt)
    m1s = csc.norm(loc = 0, scale = 1.2).rvs(size = invdt)
    m2s = csc.norm(loc = 0, scale = 1.2).rvs(size = invdt)
    m = np.array([m1s,m2s]).transpose()
    s = np.array([b1s,b2s]).transpose()
    ps = csc.uniform(loc = -1, scale = 2).rvs(size = invdt)
    cov = np.array([[b1s**2, ps * b1s * b2s],[ps*b1s*b2s,b2s**2 ]]).transpose()
    Xs = np.array([np.random.multivariate_normal(m[i],cov[i]) for i in range(invdt)]) 

    W = 0
    GG = []
    w = {k:0 for k in range(invdp)}
    for ddt in range(invdt):
        L,Q = np.linalg.eig(cov[ddt])
        M = np.matmul(Q,m[ddt])
        S = np.sqrt(np.matmul(Q**2,s[ddt]**2) + 2*cov[ddt][0,1]*Q[:,0]*Q[:,1])
        Y = (np.matmul(Q,Xs[ddt]) - M)/S
        G = sum(Y**2)
        GG.append(G)
        for ddp in range(invdp):
            if G < -2*np.log(1 - ddp*dp):
                w[ddp] += dt
    
    for ddp in range(invdp):
        W += (w[ddp] - ddp*dp)**2*dp
    return W
          

def GEN():
    DS = []
    for i in range(1000):
        if not i%50:
            print(i)
        DS.append(ES())
    for i in [5,15,25,35,50,65,75,85,95]:
        print("{}% : {}".format(100-i,np.percentile(DS,i)))

