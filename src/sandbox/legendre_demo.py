from numpy.polynomial.legendre import Legendre, legval
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#x = np.arange(-5,7.3,0.01)
#x = np.random.normal(loc=0, scale=5, size=1000)
x = np.random.uniform(low=-10, high=10, size=1000)
y = np.sin(x) + np.random.normal(scale=.1, size=x.shape)

p5 = Legendre.fit(x, y, 5)
p50 = Legendre.fit(x, y, 50)
p150 = Legendre.fit(x, y, 150)

f, ax = plt.subplots()

ax.plot(x, y, 'o', c='k', alpha=0.1)

for p, col in zip([p5, p50, p150], ['r','g','b']):
    #need to use the object (function) returned by the fitting to get
    #appropriately interpolated values
    xx, yy = np.sort(x), p(np.sort(x))
    ax.plot(xx, yy, '-o', lw=2, c=col, alpha=0.1)
    #xx, yy = p.linspace()
    #ax.plot(xx, yy, lw=2, c=col, alpha=0.7)
    print('domain:{}, window:{}'.format(p.domain, p.window))

ax.set(ylim=[-2,2])

plt.savefig('legendre_fit.pdf')
