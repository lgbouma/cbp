'''
Plot:
    (recovered depth / injected depth) vs RMS (or whatever quantity).
'''

import pandas as pd, numpy as np, matplotlib.pyplot as plt

csv = '../../results/injrecovresult/irresult_sap_top1.csv'
df = pd.read_csv(csv)

plt.close('all')
plt.style.use('classic')

f, ax = plt.subplots()

tdf, fdf = df[df['foundinj']==True], df[df['foundinj']==False]

ax.scatter(tdf['rms_biased'], tdf['depth_rec']/tdf['depth_inj'],
        lw=0, color='green', alpha=0.7, s=8)
ax.scatter(fdf['rms_biased'], fdf['depth_rec']/fdf['depth_inj'],
        lw=0, color='red', alpha=0.7, s=8)
print(min(fdf['depth_rec']))
print(min(tdf['depth_rec']))
print(min(fdf['depth_inj']))
print(min(tdf['depth_inj']))

ax.set(ylabel=r'$\delta_\mathrm{rec}/\delta_\mathrm{inj}$',
       xlabel='biased RMS (relative flux units)',
       xlim=[0,0.01],
       ylim=[-2,2],
       xscale='log')
plt.tight_layout()

savedir = '../../results/injrecovresult/plots/'
fname = 'depth_recov_vs_depth_inj_compare.pdf'
plt.savefig(savedir+fname, bbox_inches='tight')
