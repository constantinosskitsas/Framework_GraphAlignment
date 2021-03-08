import numpy as np
import scipy.sparse
from algorithms.NSD.NSD import findnz1
from algorithms.bipartiteMatching import bipartite_matching_primal_dual, bipartite_matching_setup
import math

def edge_indicator(match, ei, ej):
    ind = np.zeros(len(ei),int)
    for i in range(1,len(ei)):
        if match[ei[i]] == ej[i]:
            ind[i] = 1
    return ind


def isorank(S,w,a,b,li,lj,alpha):
    alpha = b / (a + b)
    rtype = 1
    tol = 1e-12
    maxit = 100
    verbose = True
    P = normout_rowstochastic(S)
    csum = math.fsum(w)
    v = w/csum
    n = np.shape(P)[0]
    allstats = not li and not lj
    if allstats:
        rhistsize = 5
        rp, ci, ai, tripi, m, n = bipartite_matching_setup(w,li,lj,np.amax(li),np.amax(lj))
        matm = m
        matn = n
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
        omega = math.fsum(x) - math.fsum(y)
        y = y + omega @ v
        delta = np.norm(x-y,1)#findd the correct one
        reshist[iter+1] = delta
        iter = iter + 1
        x = y * (1/math.fsum(y))
        if allstats:
            if rtype == 1:
                xf = x
            elif rtype == 2:
                xf = a*v + b/2*(S*x) #add v to preserve scale
            ai = np.zeros(len(tripi),float)#check the dimensions
            ai[tripi>0] = xf[mperm]
            _, _, _, noute1, match1= bipartite_matching_primal_dual(rp,ci,ai,matm,matn)
            ma = noute1
            mi_int = edge_indicator(match1,li,lj)#implement this
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

def normout_rowstochastic(S): #to check
    n = np.shape(S)[0]
    m = np.shape(S)[1]
    colsums = np.sum(S, dims=1)
    pi, pj, pv = findnz1(S)
    Q= scipy.csc_matrix((pv / colsums[pi], (pi, pj)), shape = (m, n))
    return Q


def netalign_setup(A,B,L,undirected):
    pass

def make_squares(A, B, L, undirected):
    m = np.shape(A)[0]
    n = np.shape(B)[0]
    if undirected:
        rpA = A.colptr
        ciA = A.rowval

        rpB = B.colptr
        ciB = B.rowval
    else:
        A = np.copy(A.conj())
        B = np.copy(B.conj())
        rpA = A.colptr#need fix
        ciA = A.rowval
        rpB = B.colptr
        ciB = B.rowval

        L = np.copy(L.conj())
        rpAB = L.colptr#
        ciAB = L.rowval#
        vAB = L.nzval#

        Se1 = []
        Se2 = []

        wv = np.zeros(n,int)
        sqi = 0
        for i in range(0,m):
        # label everything in to i in B
        # label all the nodes in B that are possible matches to the current i
            possible_matches = range(rpAB[i],rpAB[i + 1] - 1)
        # get the exact node ids via ciA[possible_matches]
            wv[ciAB[possible_matches]] = possible_matches
            for ri1 in range(rpA[i],rpA[i + 1] - 1):
        # get the actual node index
                ip = ciA[ri1]
                if i == ip:
                    continue
        # for node index ip, check the nodes that its related to in L
                for ri2 in range(rpAB[ip],rpAB[ip + 1] - 1):
                    jp = ciAB[ri2]
                    for ri3 in range(rpB[jp],rpB[jp + 1] - 1) :
                        j = ciB[ri3]
                        if j == jp:
                            continue
                        if wv[j] > 0:
            # we have a square!
                            #push!(Se1, ri2)
                            #push!(Se2, wv[j])
                            Se1.append(ri2)
                            Se2.append(wv[j])
    # remove labels for things in in adjacent to B
        wv[ciAB[possible_matches]] = 0
    Le = np.zeros((np.nnz(L), 3),int)
    LeWeights = np.zeros(np.nnz(L),float)
    for i in range(0,m):
        j = range(rpAB[i],rpAB[i + 1] - 1)
        Le[j, 1] = i
        Le[j, 2] = ciAB[j]
        LeWeights[j] = vAB[j]
    Se = np.zeros( (len(Se1), 2),int)
    Se[:, 1] = Se1
    Se[:, 2] = Se2
    return (Se, Le, LeWeights)


def netalign_setup(A, B, L, undirected):#needs fix
    Se,Le,LeWeights = make_squares(A,B,L,undirected)
    li = Le[:,1]
    lj = Le[:,2]
    Se1 = Se[:,1]
    Se2 = Se[:,2]
    S = sparse(Se1,Se2,1,nnz(L),nnz(L))

    return S,LeWeights,li,lj
