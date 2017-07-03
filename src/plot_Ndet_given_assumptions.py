# -*- coding: utf-8 -*-
'''
plot N_tra vs T_obs/P_p

plot prob_det distribution
plot prob_tra vs P_p (just to verify these look sane)
plot N_tra distribution
'''
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

import numpy as np, pandas as pd, matplotlib.pyplot as plt

workdir = '../results/N_transits_Li_2016/'
df = pd.read_csv(workdir+'Ndet_estimate_df.csv')

##################################################################
# SANITY CHECK #1: DO WE GET MORE THAN 1 TRANSIT PER CBP PERIOD? #
##################################################################
f, ax = plt.subplots(figsize=(4,4))

ax.scatter(df['T_obs']/df['P_p'], df['N_tra'], c='black', lw=0, marker='o',
        s=4, alpha=1, label='calcuated from Li+ (2016)')

x = np.linspace(0,3e3,1000)
y = x
ax.plot(x, y, c='black', lw=1, ls='-', label='one "transit" per CBP period'+\
        '\n(there should not be more than this)')

ax.set(ylabel='$N_\mathrm{tra}$',
       xlabel='$T_\mathrm{obs}/P_\mathrm{CBP}$',
       ylim=[1, max(df['N_tra']) + 0.05*max(df['N_tra'])],
       xlim=[1, max(df['T_obs']/df['P_p']) + 0.05*max(df['T_obs']/df['P_p'])],
       yscale='log',
       xscale='log')
ax.legend(loc='best', fontsize='xx-small', scatterpoints=1)

f.tight_layout()

f.savefig('../results/N_transits_Li_2016/N_tra_vs_Tobsperiodratio.pdf',
        dpi=300, bbox_inches='tight')


