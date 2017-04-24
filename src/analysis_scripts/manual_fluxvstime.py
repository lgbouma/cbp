'''
Make flux vs time to manually look thru
'''
import numpy as np, matplotlib.pyplot as plt
import pickle
import os

def make_fluxvstime(dipid):
    '''dipid = 9843451, for instance.
    '''
    ap = 'sap'

    savdir = '../../data/injrecov_pkl/real/'
    lcd = pickle.load(open(savdir+str(dipid)+'_realsearch_real.p', 'rb'))
    allq = pickle.load(open(savdir+str(dipid)+'_allq_realsearch_real.p', 'rb'))

    maxtime = max(allq['dipfind']['tfe'][ap]['times'])
    mintime = min(allq['dipfind']['tfe'][ap]['times'])

    # Set up matplotlib figure and axes.
    plt.close('all')
    f, ax = plt.subplots(figsize=(16, 5))

    for qnum in np.sort(list(lcd.keys())):

        min_inum = np.min(list(lcd[qnum]['white'].keys()))

        # If u want raw
        #lc = lcd[qnum]['dtr'][ap]
        #times = lc['times']
        #fluxs = lc['fluxs_dtr_norm']
        #errs = lc['errs_dtr_norm']

        # If u want subtracted
        max_inum = np.max(list(lcd[qnum]['white'].keys()))
        #min_inum = np.min(list(lcd[qnum]['white'].keys()))
        lc = lcd[qnum]['white'][max_inum][ap]['legdict']['whiteseries']
        times = lc['times']
        fluxs = lc['wfluxsresid']
        errs = lc['errs']

        # Plot the same thing three times.
        ax.plot(times, fluxs, c='k', linestyle='-',
                marker='o', markerfacecolor='k',
                markeredgecolor='k', ms=1, lw=0.1)

    f.tight_layout()

    plt.show()


if __name__ == '__main__':
    dipid = 9480977
    make_fluxvstime(dipid)
