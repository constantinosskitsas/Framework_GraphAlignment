import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
#df=pd.read_excel("hi.xlsx")
df=pd.read_excel("gp.xlsx","iso")
#plt.plot(df["CONE",],df["GRASP"],df["REGAL"],df["LREA"],df["NSD"],df["Isorank"])
#plt.plot(df["Noise"],df["GWL"],"x-c", label="GWL")
plt.plot(df["noise"],df["NN-A"],"o-g", label="NN-A")
plt.plot(df["noise"],df["SG-A"],"v-r", label="SGC-A")
plt.plot(df["noise"],df["JV-A"],"*-b", label="JV-A")
plt.plot(df["noise"],df["NN-PL"],"o--g", label="NN-PL")
plt.plot(df["noise"],df["SG-PL"],"v--r", label="SG-PL")
plt.plot(df["noise"],df["JV-PL"],"*--b", label="JV-PL")
#plt.xlabel("Noise Level")
plt.ylim([0, 1.1])
#plt.ylabel("Accuracy")
plt.title("Isorank")
plt.legend()
#how to save legend
#legend = plt.legend(mode="expand",loc="best",ncol=7,fontsize="small")
#fig  = legend.figure
#fig.canvas.draw()
#bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#fig.savefig("hi.png", dpi="figure", bbox_inches=bbox)
plt.margins(x=0)
plt.savefig("Isorank.pdf",bbox_inches='tight')

plt.show()



