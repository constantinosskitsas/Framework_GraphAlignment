
import numpy as np
from data import ReadFile

gt="data/noise_level_1/gt_1.txt"
gmb, gma = ReadFile.gt1(gt)
gma= gma.astype(int)
gmb=gmb.astype(int)
gma1=np.zeros(len(gma))
gmb1=np.zeros(len(gma))
for i in range(len(gma)):
    gma1[i]=i
    gmb1[gma[i]]=gmb[i]
print(gmb1)
print(gma1)
print(gmb)
print(gma)


