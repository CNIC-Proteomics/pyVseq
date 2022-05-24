# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:10:18 2022

@author: alaguillog
"""
import math
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pandas as pd

df = pd.read_csv(r"venn.txt", sep="\t")

df.columns = ["Row Labels", "error2", "error12"]
df["A"] = df.apply(lambda x: False if math.isnan(x.error2) else True, axis=1)
df["B"] = df.apply(lambda x: False if math.isnan(x.error12) else True, axis=1)
df["C"] = df.apply(lambda x: True if x.A == True and x.B == True else False, axis=1)

df["A"] = df.apply(lambda x: False if x.C == True else x.A, axis=1)
df["B"] = df.apply(lambda x: False if x.C == True else x.B, axis=1)

len(df[df.A==True])
len(df[df.B==True])
len(df[df.C==True])

venn2(subsets = (len(df[df.A==True]), len(df[df.B==True]), len(df[df.C==True])), set_labels = ('Error 0.02', 'Error 0.12'))
plt.show()

fig = plt.figure()
fig.set_size_inches(10, 10)
total = len(df[df.A==True]) + len(df[df.B==True]) + len(df[df.C==True])
venn2(subsets = (len(df[df.A==True]), len(df[df.B==True]), len(df[df.C==True])), set_labels = ('Error 0.02', 'Error 0.12'),
      subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total):1.0%}" + ")")
plt.tight_layout()
fig.savefig(os.path.join(Path(outpath), outgraph))
