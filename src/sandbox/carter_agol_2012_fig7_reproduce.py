'''
reproduce top panel of figure 7 of Carter and Agol 2012
'''

# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.preamble': [
        '\\usepackage{amsmath}',
        '\\usepackage{amssymb}'
        ]
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import sys
sys.path.insert(1, '../../../qatspy/') # forgive me
import qatspy
import numpy as np, matplotlib.pyplot as plt

flux = np.genfromtxt('../../../qatspy/kep36.dat')
N = int(len(flux))

Δ_min_arr = np.arange(350, 1300, 1)
#Δ_min_arr = np.arange(794, 796, 1)

q_arr = np.arange(3, 15, 1)

f, axs = plt.subplots(nrows=2, ncols=1, figsize=(5,4))

for c, diff in zip(['k','r','g'],[0, 1, 2]):

    S_best_l, M_best_l, q_best_l = [], [], []

    for Δ_min in Δ_min_arr:
        if Δ_min % 10 == 0:
            print(Δ_min)

        Δ_max = Δ_min + diff

        S_, M_ = [], []
        for q in q_arr:

            d = qatspy.shConvol_wrapper(flux, q)

            result = qatspy.qats_wrapper(d, Δ_min, Δ_max, N, q)

            S_.append(result[0])
            M_.append(result[1])

        S_best = np.max(S_)
        M_best = M_[np.argmax(S_)]
        q_best = q_arr[np.argmax(S_)]

        S_best_l.append(S_best)
        M_best_l.append(M_best)
        q_best_l.append(q_best)

    axs[0].plot(Δ_min_arr, S_best_l, color=c, lw=1,
            label='$\Delta_\mathrm{max} - \Delta_\mathrm{min} = $'+\
                  '{:d}'.format(diff))
    axs[1].plot(Δ_min_arr, S_best_l, color=c, lw=1,
            label='$\Delta_\mathrm{max} - \Delta_\mathrm{min} = $'+\
                  '{:d}'.format(diff))


axs[0].set_title('fig 7 of Carter \& Agol (2012): QATS detects Kepler 36c',
        fontsize='small')
axs[0].set_ylabel('max total transit S/N')
axs[1].set_ylabel('max total transit S/N')
axs[1].set_xlabel('$\Delta_\mathrm{min}$ (units of Kepler cadence count; top spectrum is 7.3d - 27d)',
        fontsize='small')
axs[0].set_xlim([min(Δ_min_arr), max(Δ_min_arr)])
axs[1].legend(loc='best', fontsize='small')

axs[1].set_xlim([780, 830])

f.tight_layout()
f.savefig('carter_agol_2012_fig7_reproduce.pdf', bbox_inches='tight')
