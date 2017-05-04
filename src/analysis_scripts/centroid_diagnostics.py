'''
In results/real_search/dip_deepdive, plot (vector distance from median of
centroid) vs time, for all quarters. (The plots are column # vs row #). Points
close to the expected transit time are highlighted with vertical lines.
'''
import numpy as np, matplotlib.pyplot as plt
import pickle
import os
import pdb

def plot_centroid_diagnostic(dipid, t_0, period):
    ap = 'sap'
    savdir = '../../data/injrecov_pkl/real/'
    lcd = pickle.load(open(savdir+str(dipid)+'_realsearch_real.p', 'rb'))
    allq = pickle.load(open(savdir+str(dipid)+'_allq_realsearch_real.p', 'rb'))

    maxtime = max(allq['dipfind']['tfe'][ap]['times'])
    mintime = min(allq['dipfind']['tfe'][ap]['times'])

    transit_times = np.array(list(range(-100,100)))*period + t_0
    transit_times = transit_times[(transit_times < maxtime) & (transit_times >
        mintime)]

    colors = ['r', 'g', 'b', 'gray']
    plt.close('all')
    f, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=1)

    for qnum in np.sort(list(lcd.keys())):

        #"moment-derived column centroid"
        centroid_x = lcd[qnum]['mom_centr1']
        centroid_y = lcd[qnum]['mom_centr2']
        r = np.sqrt(centroid_x**2 + centroid_y**2)
        r0 = np.median(r[np.isfinite(r)])
        r_minus_r0 = r-r0

        min_inum = np.min(list(lcd[qnum]['white'].keys()))
        max_inum = np.max(list(lcd[qnum]['white'].keys()))
        lc = lcd[qnum]['white'][max_inum][ap]['legdict']['whiteseries']
        times = lc['times']
        fluxs = lc['wfluxs']
        errs = lc['errs']

        thiscolor = colors[int(qnum)%len(colors)]
        # centroid vs time
        axs[0].plot(lcd[qnum]['time'], r_minus_r0, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=1, lw=0.1)
        # flux vs time
        axs[1].plot(times, fluxs, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=1, lw=0.1)

    for ax in axs:
        ylims = ax.get_ylim()
        ax.vlines(transit_times, min(ylims), max(ylims), colors='gray',
                linestyles='solid', zorder=-5, alpha=0.3)
        ax.set(ylim=ylims)
    axs[1].set(xlabel='time [days]', ylabel='relative flux',
           title='constant {:.4g} period centered on transits'.format(period))
    axs[0].set(ylabel='r-r_0 [pixels, r_0 is median centroid per quarter]')

    f.tight_layout()

    savedir = '../../results/real_search/dip_deepdive/centroid_diagnostic/'
    f.savefig(savedir+str(dipid)+'.png',dpi=350)

if __name__ == '__main__':
    id_t0_period = [(9480977,158.34006,38.04367),#check this t0 w/ manual measurements
                    (6791604,144.79417,18.43698),#same
                    (7889628,155.07355,24.2372)]

    for dipid, t_0, period in id_t0_period:
        plot_centroid_diagnostic(dipid, t_0, period)
        print('made {:s}'.format(str(dipid)))
