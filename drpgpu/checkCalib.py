import numpy as np
import matplotlib.pyplot as plt
with open("data/cxid9114_r95_evt01_calib.txt",'r') as f: known = f.read().splitlines()
with open("calc_calib.txt",'r') as f: calc = f.read().splitlines()

knownData = np.array(known).astype('double')
calcData = np.array(calc).astype('double')
diffData = knownData - calcData

for i, val in enumerate(diffData):
  if abs(val) > 5.0: print i, val, knownData[i], calcData[i]

plt.subplot(2,1,1)
plt.plot(knownData, 'r', label='Known Data')
plt.plot(calcData, 'b', label='Calcl Data')
plt.legend()
plt.subplot(2,1,2)
plt.plot(diffData, 'g', label='Differences')
plt.legend()
plt.show()

