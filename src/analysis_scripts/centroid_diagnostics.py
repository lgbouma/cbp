'''
In results/real_search/dip_deepdive, make 2x2 grid of centroid timeseries. (The
plots are column # vs row #). The points are colored by proximity to expected
transit time.
'''
import numpy as np, matplotlib.pyplot as plt
import pickle
import os

def centroid_diagnostics(dipid):
    '''e.g., dipid = 9843451
    '''
    ap = 'sap'
    savdir = '../../data/injrecov_pkl/real/'
    lcd = pickle.load(open(savdir+str(dipid)+'_realsearch_real.p', 'rb'))
    allq = pickle.load(open(savdir+str(dipid)+'_allq_realsearch_real.p', 'rb'))

    #FIXME: actually write this!
    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, max(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]

    # assume BLS actually found the right period...
    foldperiod = nbestperiods[0]
    fbls = allq['dipfind']['bls'][ap]['finebls'][foldperiod]
    plotφ = fbls['φ']
    plotfluxs = fbls['flux_φ']
    binplotφ = fbls['binned_φ']
    binplotfluxs = fbls['binned_flux_φ']


    maxtime = max(allq['dipfind']['tfe'][ap]['times'])
    mintime = min(allq['dipfind']['tfe'][ap]['times'])

    colors = ['r', 'g', 'b', 'gray']
    # Set up matplotlib figure and axes.
    plt.close('all')
    f, axs = plt.subplots(figsize=(16, 10), nrows=2, ncols=2)

    for qnum in np.sort(list(lcd.keys())):

        centroid_x = lcd[qnum]['mom_centr1']
        centroid_y = lcd[qnum]['mom_centr2']




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
