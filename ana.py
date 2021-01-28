import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

v = defaultdict(list)

d = iter(Path("run.log").read_text().split('\n'))
for l in d:
    try:
        v[l.split()[0]].append(float(next(d)))
    except:
        break

ts = []
V = []
ΔV = []
for t, vs in v.items():
    ts.append(float(t))
    V.append(np.mean(vs))
    ΔV.append(np.std(vs) / np.sqrt(len(vs)))

plt.errorbar(ts, V, yerr=ΔV, capsize=1.0, linestyle="None", marker="x")
plt.xscale("log")
plt.xlabel("T")
plt.ylabel("binding rate")
plt.show()
