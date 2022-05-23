import numpy as np
#from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv
from numpy.linalg import eigh,eig
import networkx as nx 
import random
#from lapsolver import solve_dense



def seigh(A):
  """
  Sort eigenvalues and eigenvectors in descending order. 
  Not used.
  """
  l, u = np.linalg.eigh(A)
  idx = l.argsort()[::-1]   
  l = l[idx]
  u = u[:,idx]
  return l, u
def main(data, eta):
  Src = data['Src']
  Tar = data['Tar']
  n = Src.shape[0]
  l,U =eigh(Src)
  mu,V = eigh(Tar)
  l = np.array([l])
  mu = np.array([mu])

  #Eq.4
  coeff = 1.0/((l.T - mu)**2 + eta**2)
  #Eq. 3
  coeff = coeff * (U.T @ np.ones((n,n)) @ V)
  X = U @ coeff @ V.T

  Xt = X.T
  Xt=X
  # Solve with linear assignment maximizing the similarity 
  # row,col = linear_sum_assignment(Xt, maximize=True)

  # Alternatively, we can use a more efficient solver.
  # The solver works on cost minimization, so take -X 
  #rows, cols = solve_dense(-Xt)
  #return rows, cols 
  return Xt

def grampa(Src, Tar, eta):
  """
  Summary or Description of the Function

  Parameters:
  Src (np.array): The nxn adjacency matrix of the first graph 
  Tar (np.array): The nxn adjacency matrix of the second graph
  eta (float): The eta value of Eq. 4 in the paper

  Returns:
  Xt similarity Matrix
  """
  n = Src.shape[0]
  l,U = eigh(Src)
  mu,V = eigh(Tar)
  l = np.array([l])
  mu = np.array([mu])

  #Eq.4
  coeff = 1.0/((l.T - mu)**2 + eta**2)
  #Eq. 3
  coeff = coeff * (U.T @ np.ones((n,n)) @ V)
  X = U @ coeff @ V.T

  Xt = X.T
  # Solve with linear assignment maximizing the similarity 
  # row,col = linear_sum_assignment(Xt, maximize=True)

  # Alternatively, we can use a more efficient solver.
  # The solver works on cost minimization, so take -X 
  #rows, cols = solve_dense(-Xt)
  #return rows, cols 
  return Xt