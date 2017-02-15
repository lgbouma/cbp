'''
When the fit is done, the domain is first mapped to the window by a linear
transformation and the usual least squares fit is done using the mapped data
points. The window and domain of the fit are part of the returned series and
are automatically used when computing values, derivatives, and such. If they
aren’t specified in the call the fitting routine will use the default window
and the smallest domain that holds all the data points. This is illustrated
below for a fit to a noisy sine curve.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T
np.random.seed(11)

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x) + np.random.normal(scale=.1, size=x.shape)
p = T.fit(x, y, 5)
plt.plot(x, y, 'o')

xx, yy = p.linspace()
plt.plot(xx, yy, lw=2)

print('domain:{}, window:{}'.format(p.domain, p.window))


plt.savefig('cheby_fit.pdf')
