'''
In results/real_search/dip_deepdive, make easier to read lightcurves, with 3
rows and flux vs time split appropriately.
'''
import numpy as np, matplotlib.pyplot as plt
import pickle
import os

def make_fluxvstime_panel(dipid):
    '''dipid = 9843451, for instance.
    '''
    ap = 'sap'

    savdir = '../../data/injrecov_pkl/real/'
    lcd = pickle.load(open(savdir+str(dipid)+'_realsearch_real.p', 'rb'))
    allq = pickle.load(open(savdir+str(dipid)+'_allq_realsearch_real.p', 'rb'))

    maxtime = max(allq['dipfind']['tfe'][ap]['times'])
    mintime = min(allq['dipfind']['tfe'][ap]['times'])

    nrows = 4
    colors = ['r', 'g', 'b', 'gray']

    # Set up matplotlib figure and axes.
    plt.close('all')
    f, axs = plt.subplots(figsize=(16, 10), nrows=nrows, ncols=1)

    for qnum in np.sort(list(lcd.keys())):

        min_inum = np.min(list(lcd[qnum]['white'].keys()))
        lc = lcd[qnum]['white'][min_inum][ap]['legdict']['whiteseries']

        times = lc['times']
        fluxs = lc['wfluxs']
        errs = lc['errs']

        thiscolor = colors[int(qnum)%len(colors)]

        # Plot the same thing three times.
        for ax in axs:
            ax.plot(times, fluxs, c=thiscolor, linestyle='-',
                    marker='o', markerfacecolor=thiscolor,
                    markeredgecolor=thiscolor, ms=0.1, lw=0.1)

        # Now fix the xlims so it looks like 3 different plots.
        panel_timelen = (maxtime-mintime)/nrows
        xlims = []
        for i in range(0, nrows):
            xlims.append(
                    (mintime+i*panel_timelen, mintime+(i+1)*panel_timelen)
                    )

        for ix, ax in enumerate(axs):
            ax.set_xlim(xlims[ix])


    f.tight_layout()

    savedir = '../../results/real_search/dip_deepdive/fluxvstime_panels/'
    f.savefig(savedir+str(dipid)+'.png',dpi=300)

if __name__ == '__main__':
    dipids = [int(f.split('.')[0]) for f in \
            os.listdir('../../results/real_search/dip_deepdive/') if \
            f.endswith('.txt')]

    for dipid in dipids:
        make_fluxvstime_panel(dipid)
        print('made {:s}'.format(str(dipid)))
