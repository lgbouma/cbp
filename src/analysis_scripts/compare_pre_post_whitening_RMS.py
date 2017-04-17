'''
cumulative histogram of biased RMS once whitened for
    1. only a single iteration of whitening
    2. iterative whitening
(facilitates their comparison)
'''
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns

csv0 = '../../results/injrecovresult/success0/irresult_sap_top1.csv'
csv1 = '../../results/injrecovresult/success1/irresult_sap_top1.csv'
df0 = pd.read_csv(csv0)
df1 = pd.read_csv(csv1)

plt.close('all')
plt.style.use('classic')

ax = sns.distplot(df0['rms_biased'], kde=False, norm_hist=True,
         label='first whitening step only'+\
               '\nN={:d}'.format(int(len(df0))),
         hist_kws={'cumulative':True,
                   'histtype':'step',
                   'linewidth':4,
                   'color': 'g'})

ax = sns.distplot(df1['rms_biased'], kde=False, norm_hist=True,
         label='iterative whitening,\nnwhitenmax=8,\nrmsfloor=5e-4 (0.05%)'+\
               '\nN={:d}'.format(int(len(df1))),
         hist_kws={'cumulative':True,
                   'histtype':'step',
                   'linewidth':4,
                   'color': 'b'})

ax.set(xscale='log', ylim=[0,1], ylabel='cumulative fraction',
      xlabel='biased RMS once whitened (relative flux units)')
ax.legend(loc='upper left', fontsize='x-small')
plt.tight_layout()

savedir = '../../results/injrecovresult/plots/'
fname = 'cdf_whitening_methods_compare.pdf'
plt.savefig(savedir+fname, bbox_inches='tight')
