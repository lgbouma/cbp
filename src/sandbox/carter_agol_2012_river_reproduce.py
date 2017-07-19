'''
reproduce bottom panel of figure 7 of Carter and Agol 2012.

aka the RIVER PLOT
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

import qatspy
import numpy as np, matplotlib.pyplot as plt

flux = np.genfromtxt('../../../qatspy/kep36.dat')
N = int(len(flux))

#Δ_min_arr = np.arange(350, 1300, 1)
#Δ_min_arr = np.arange(794, 795, 1)
Δ_min_arr = np.arange(793, 794, 1)

q_arr = np.arange(14, 15, 1)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,5))

#for diff in [0, 1, 2]:
for diff in [2]:

    S_best_l, M_best_l, q_best_l, δ_best_l, inds_l = [], [], [], [], []

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
        δ_best = S_best / np.sqrt(M_best * q_best)

        inds = qatspy.qats_indices_wrapper(d, M_best, Δ_min, Δ_max, N, q_best)

        S_best_l.append(S_best)
        M_best_l.append(M_best)
        q_best_l.append(q_best)
        δ_best_l.append(δ_best)

        inds_l.append(inds)


    intperiod = 794
    period, t_0 = 794.36, 120
    ind_fold = np.mod(inds, period) - t_0

    t_fold = np.mod(np.arange(N), period) - t_0

    xvals = np.arange(-20, 20+1, 1)

    # get the flux values
    flux_arr = np.zeros((len(xvals)-1, M_best-1 ))

    for tra_ind in range(M_best-1):

        begin = int(np.floor(tra_ind*period)) + t_0 + xvals[0]
        end = int(np.floor(tra_ind*period)) + t_0 + xvals[-1]

        flux_arr[:, tra_ind] = flux[begin:end]

    # for plotting, want flux_arr 0 values to be min for colorscale
    flux_arr[flux_arr == 0] = np.max(flux_arr)

    im = ax.pcolor(xvals,
                   list(range(M_best)),
                   flux_arr.T,
                   cmap='Blues_r')
                   #cmap='Greys_r') #looks alright too

    ax.plot(ind_fold[:M_best-1], np.array(range(M_best-1))+0.5, ls='-',
            c='red', lw=0, marker='o', markerfacecolor='red',
            markeredgecolor='red', ms=2)

ax.set_title('$\Delta_\mathrm{max} - \Delta_\mathrm{min} = $'+\
        '{:d}'.format(diff), fontsize='small')
ax.set_ylabel('transit number')
ax.set_xlabel('(cadence number mod 794.36) - 120',
        fontsize='small')

ax.set_ylim([0, M_best-1-1])
ax.set_xlim([-20, 20])

f.tight_layout()
f.savefig('carter_agol_2012_river.pdf', bbox_inches='tight', dpi=300)
