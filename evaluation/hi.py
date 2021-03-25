import numpy as np 
from sklearn.metrics import jaccard_score
from numpy.linalg import inv
print ("Hello World!")
A=np.random.randint(2, size=(5, 5))
B=np.random.randint(2, size=(5, 5))
print (B)
Gt = np.random.permutation(5)
Tar_e = Gt[B]
print (Tar_e)
JI=0
new=np.zeros(5)
for i in range(5):
    new[Gt[i]]=i
new1=new[Tar_e]
print (new1)
permuted = Gt[Tar_e]
print (permuted)
for i in range(5):
        #print(jaccard_score(A[i,:],B[mb[i],:]))
        #JI=JI+jaccard_score(A[i,:],B[mb[i],:])
    JI=JI+jaccard_score(B[i,:],new1[i,:],average='micro')
print (JI)

