from matplotlib import pyplot as plt
import numpy as np
t=np.array(range(0,256))
a=3
per=40
noise=np.random.normal(np.mean(a*np.sin(2*np.pi/per*t)), 1, len(t))
s=a*np.sin(2*np.pi/per*t)+noise
s_pure=a*np.sin(2*np.pi/per*t)
plt.plot(t, s)
plt.plot(t, s_pure)
plt.show()