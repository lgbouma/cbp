'''
Make flux vs time to manually look thru
'''
import numpy as np, matplotlib.pyplot as plt
import pickle
import os

def make_fluxvstime(dipid):
    ap = 'sap'

    savdir = '../../data/injrecov_pkl/real/'
    lcd = pickle.load(open(savdir+str(dipid)+'_realsearch_real.p', 'rb'))
    allq = pickle.load(open(savdir+str(dipid)+'_allq_realsearch_real.p', 'rb'))

    maxtime = max(allq['dipfind']['tfe'][ap]['times'])
    mintime = min(allq['dipfind']['tfe'][ap]['times'])

    t_0 = 144.79417
    period = 18.43698
    transit_times = np.array(list(range(-150,150)))*period + t_0
    transit_times = transit_times[(transit_times < maxtime) & (transit_times >
        mintime)]

    # Set up matplotlib figure and axes.
    plt.close('all')
    f, ax = plt.subplots(figsize=(16, 5))

    for qnum in np.sort(list(lcd.keys())):

        min_inum = np.min(list(lcd[qnum]['white'].keys()))

        # If u want raw
        lc = lcd[qnum]['dtr'][ap]
        times = lc['times']
        fluxs = lc['fluxs_dtr_norm']
        errs = lc['errs_dtr_norm']

        # If u want subtracted
        #max_inum = np.max(list(lcd[qnum]['white'].keys()))
        ##min_inum = np.min(list(lcd[qnum]['white'].keys()))
        #lc = lcd[qnum]['white'][max_inum][ap]['legdict']['whiteseries']
        #times = lc['times']
        #fluxs = lc['wfluxsresid']
        #errs = lc['errs']

        mask = ~((times > 197) & (times < 203))
        # Plot the same thing three times.
        ax.plot(times[mask], fluxs[mask], c='k', linestyle='-',
                marker='o', markerfacecolor='k',
                markeredgecolor='k', ms=1, lw=0.1)

    f.tight_layout()
    ylims = ax.get_ylim()
    ax.vlines(transit_times, min(ylims), max(ylims), colors='gray',
            linestyles='solid', zorder=-5, alpha=0.3)
    #ax.set(xlim=[139.2,1580.97], ylim=[-0.001332,0.00044483])
    ax.set(xlabel='time [days]', ylabel='relative flux',
           title='constant {:.4g} period centered on transits'.format(period))

    #f.savefig('../../results/real_search/dip_deepdive/manual_fluxvstime/7889628_raw_0.png',
    #        dpi=300)
    plt.show()

if __name__ == '__main__':
    dipid = 7889628
    make_fluxvstime(dipid)
