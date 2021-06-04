
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from scipy.stats import poisson
import collections
def plotG(G, name="", end=True, circular=False):

    G = nx.Graph(G)

    plt.figure(name)

    if len(G) <= 200:
        kwargs = {}
        if circular:
            kwargs = dict(pos=nx.circular_layout(G),
                          node_color='r', edge_color='b')
        plt.subplot(211)
        nx.draw(G, **kwargs)

        plt.subplot(212)

    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt, width=0.80, color="b")

    # print(degreeCount)
    plt.title(
        f"{name} Degree Histogram.\nn = {len(G)}, e = {len(G.edges)}, maxd = {deg[0]}, disc = {degreeCount[0]}")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # fig, ax = plt.subplots()
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)

    plt.show(block=end)


#kdistr= np.random.randint(low=6,high=10,size=1000)
kdistr=[]
for i in range (0,1000):
    kdistr.append(random.randint(15, 21))
#powerl=np.random.power(5, size=1000)
powerl = nx.utils.powerlaw_sequence(1000, 2.5)
normald=np.random.normal(10,1,1000)
#poissond = poisson.rvs(mu=10, size=1000)
#poissond=np.random.poisson(lam=10, size=1000)
#kdistr = [int(num) for num in kdistr]
#powerl = [round(num) for num in powerl]
normald = [round(num) for num in normald]
#poissond = [round(num) for num in poissond]
#for i in len(powerl):
#    if powerl[i]<5:
#        powerl[i]=powerl[i]*3
#ksum=sum(kdistr)
#psum=sum(powerl)
usum=sum(normald)
#p1sum=sum(poissond)
# if ksum%2==1:
#     print("hi")
#     max_value = max(kdistr)
#     max_index = kdistr.index(max_value)
#     kdistr[max_index]=kdistr[max_index]-1
# if psum%2==1:
#     max_value = max(powerl)
#     max_index = powerl.index(max_value)
#     powerl[max_index]=powerl[max_index]-1
if usum%2==1:
    max_value = max(normald)
    max_index = normald.index(max_value)
    normald[max_index]=normald[max_index]-1
# if p1sum%2==1:
#     max_value = max(poissond)
#     max_index = poissond.index(max_value)
#     poissond[max_index]=poissond[max_index]-1

# G = nx.configuration_model(kdistr)
# G.remove_edges_from(nx.selfloop_edges(G))
# G1 = nx.configuration_model(powerl)
# G1.remove_edges_from(nx.selfloop_edges(G1))
G2 = nx.configuration_model(normald)
G2 = nx.Graph(G2)
G2.remove_edges_from(nx.selfloop_edges(G2))
#G3 = nx.configuration_model(poissond)
# G3.remove_edges_from(nx.selfloop_edges(G3))
# sizes = [3000, 4500, 2000,500]
# probs = [[0.01, 0.003, 0.002,0.002], [0.003, 0.03, 0.005,0.002], [0.002, 0.005, 0.04,0.002], [0.002, 0.002, 0.002, 0.02]]
# g = nx.stochastic_block_model(sizes, probs, seed=0)
#plotG(g)
#plotG(G)
plotG(G2)
#plotG(G2)
#plotG(G3)