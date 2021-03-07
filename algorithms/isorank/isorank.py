import numpy as np

from algorithms.bipartiteMatching import bipartite_matching_primal_dual, bipartite_matching_setup


def edge_indicator(M_output, li, lj):
    pass


def isorank(S,w,a,b,li,lj,alpha):
    alpha = b / (a + b)
    rtype = 1
    tol = 1e-12
    maxit = 100
    verbose = True
    P = normout_rowstochastic(S)
    csum = sum_kbn(w)
    v = w/csum
    n = np.shape(P)[0]
    allstats = not li and not lj
    if allstats:
        rhistsize = 5
        M_setup = bipartite_matching_setup(w,li,lj,np.amax(li),np.amax(lj))
        tripi = M_setup.tripi
        matm = M_setup.m
        matn = M_setup.n
        rp = M_setup.rp
        ci = M_setup.ci
        mperm = tripi[tripi>0] # a permutation for the matching problem
    else:
        rhistsize = 1
    r = alpha
    x = np.zeros(n,float) + v
    delta = 2
    iter = 0
    reshist = np.zeros((maxit,rhistsize),float)
    xbest = x
    fbest = 0
    fbestiter = 0
    if verbose and allstats: #print the header
        print("%5s   %4s   %8s   %7s %7s %7s %7s\n","best","iter","pr-delta","obj","weight","card","overlap")
    elif verbose:
        print("%4s   %8s\n","iter","delta")
    while iter < maxit and delta > tol:
        y = r @ (P.conj() @ x)
        omega = sum_kbn(x) - sum_kbn(y)
        y = y + omega @ v
        delta = np.norm(x-y,1)#findd the correct one
        reshist[iter+1] = delta
        iter = iter + 1
        x = y * (1/sum_kbn(y))
        if allstats:
            if rtype == 1:
                xf = x
            elif rtype == 2:
                xf = a*v + b/2*(S*x) #add v to preserve scale
            ai = np.zeros(len(tripi),float)#check the dimensions
            ai[tripi>0] = xf[mperm]
            M_output = bipartite_matching_primal_dual(rp,ci,ai,matm,matn)
            ma = M_output.cardinality
            mi_int = edge_indicator(M_output,li,lj)#implement this
            val = np.dot(w,mi_int)
            overlap = np.dot(mi_int,(S*mi_int)/2)
            f = a*val + b*overlap
            if f > fbest:
                xbest = x
                fbest = f
                fbestiter = iter
                itermark = "*"
            else:
                itermark = " "
            if verbose and allstats:
                print("%5s   %4i   %8.1e   %5.2f %7.2f %7i %7i\n", itermark, iter, delta, f, val, ma, overlap)
                reshist[iter,2:-1] = [a*val + b*overlap, val, ma, overlap]
            elif verbose:
                print("%4i    %8.1e\n", iter, delta)
    flag = delta>tol
    reshist = reshist[1:iter,:]
    if allstats:
        x=xbest
    return x,flag,reshist

def normout_rowstochastic(S):
    pass

def sum_kbn(w):
    pass