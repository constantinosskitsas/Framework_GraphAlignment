import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
#df=pd.read_excel("hi.xlsx")
df=pd.read_excel("G2.xlsx","n3s")
plt.plot(df["Noise"],df["Gwl"],"x-c", label="GWL")
plt.plot(df["Noise"],df["Cone"],"o-g", label="Cone")
plt.plot(df["Noise"],df["Grasp"],"v-r", label="Grasp")
plt.plot(df["Noise"],df["Regal"],"s-k", label="Regal")
plt.plot(df["Noise"],df["Lrea"],"D-m", label="Lrea")
plt.plot(df["Noise"],df["NSD"],"p-y", label="Nsd")
plt.plot(df["Noise"],df["Isorank"],"*-b", label="Isorank")
#plt.xlabel("% edges")
plt.ylim([0, 1.1])
#plt.xticks(df["Noise"])
#plt.ylabel("Accuracy")
plt.title("Watts Strogats random graph")
#plt.legend()()
#how to save legend
#legend = plt.legend(mode="expand",loc="best",ncol=7,fontsize="small")
##fig  = legend.figure
#fig.canvas.draw()
#bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#fig.savefig("hi.png", dpi="figure", bbox_inches=bbox)
plt.margins(x=0)
plt.savefig("G2-n3s.pdf",bbox_inches='tight')
plt.show()


