'''
The "dipsearchplots" and "whitened_diagnostics" made immediately from the
pipeline are insufficient to rule out many cases.
'''
import pdb
import matplotlib
matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import logging
from datetime import datetime

import inj_recov as ir
import inj_recov_plots as irp

import os

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.kepler' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )

###############################################################################
###############################################################################

def vet_bls(lcd, allq, ap='sap', inj=False, varepoch='bls',
        phasebin=0.002, inset=True):
    '''
    dipsearchplot, improved.

    |           flux vs time for all quarters              |
    | phased 1     |    phased 2      |       phased 3     |
    | phased 4     |    phased 5      |       periodogram  |

    Args:
    lcd (dict): dictionary with all the quarter-keyed data.

    allq (dict): dictionary with all the stitched full-time series data.

    ap (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    varepoch (str or None): if 'bls', goes by the ingress/egress computed in
        binned-phase from BLS.

    Returns: nothing, but saves the plot with a smart name to
        ../results/dipsearchplot/
    '''

    assert ap == 'sap' or ap == 'pdc'

    kicid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    LOGINFO('beginning dipsearch plot, KIC {:s}'.format(
        str(kicid)))

    # Set up matplotlib figure and axes.
    plt.close('all')
    nrows, ncols = 3, 3
    f = plt.figure(figsize=(16, 10))
    gs = GridSpec(nrows, ncols) # 5 rows, 5 columns

    # row 0: SAP/PDC timeseries
    ax_raw = f.add_subplot(gs[0,:])
    # phased at 5 favorite periods
    axs_φ = []
    for i in range(1,3):
        for j in range(0,ncols):
            if i == 2 and j == 2:
                continue
            else:
                axs_φ.append(f.add_subplot(gs[i,j]))
    # periodogram
    ax_pg = f.add_subplot(gs[2,2])

    f.tight_layout(h_pad=-0.7)

    ###############
    # TIME SERIES #
    ###############
    qnums = np.unique(allq['dipfind']['tfe'][ap]['qnums'])
    lc = allq['dipfind']['tfe'][ap]
    quarters = lc['qnums']
    MAD = np.median( np.abs( lc['fluxs'] - np.median(lc['fluxs']) ) )
    ylim_raw = [np.median(lc['fluxs']) - 5.5*(1.48*MAD),
                np.median(lc['fluxs']) + 2.5*(1.48*MAD)]

    for ix, qnum in enumerate(qnums):

        times = lc['times'][quarters==qnum]
        fluxs = lc['fluxs'][quarters==qnum]
        errs = lc['errs'][quarters==qnum]

        ax_raw.plot(times, fluxs, c='k', linestyle='-',
                marker='o', markerfacecolor='k',
                markeredgecolor='k', ms=3, lw=0., zorder=1,
                rasterized=True)
        ax_raw.plot(times, fluxs, c='orange', linestyle='-',
                marker='o', markerfacecolor='orange',
                markeredgecolor='orange', ms=1.5, lw=0., zorder=2,
                rasterized=True)

        txt = '%d' % (int(qnum))
        txt_x = np.nanmin(times) + (np.nanmax(times)-np.nanmin(times))/2
        txt_y = np.median(fluxs) + np.nanstd(fluxs)
        if txt_y < ylim_raw[0]:
            txt_y = np.nanmin(ylim_raw) + 0.001
        txt_y = min(txt_y, max(ylim_raw))

        t = ax_raw.text(txt_x, txt_y, txt, horizontalalignment='center',
                verticalalignment='center', fontsize=6, zorder=3)
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='black'))

        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = np.nanmin(times)
            max_time = np.nanmax(times)
        elif ix > 0:
            if np.nanmin(times) < min_time:
                min_time = np.nanmin(times)
            if np.nanmax(times) > max_time:
                max_time = np.nanmax(times)

    # Get transit ephemeris
    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, np.nanmax(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]
    if len(nbestperiods)==0:
        LOGERROR('{:s} has no good periods...'.format(kicid))
        raise AssertionError
    foldperiod, count = 0, 0
    # Only show periods > 10 days, otherwise it's overlapping and not useful
    while foldperiod < 10:
        try:
            foldperiod = nbestperiods[count]
        except:
            foldperiod = False
            break
        count += 1

    if foldperiod:
        fbls = allq['dipfind']['bls'][ap]['finebls'][foldperiod]
        φ_0 = fbls['φ_0']
        t_0 = min_time + φ_0*foldperiod
        ephem_times = t_0 + foldperiod * np.arange(0, (max_time-t_0)/foldperiod, 1)

        # plot ephemeris
        ax_raw.plot(ephem_times,0.02*(max(ylim_raw)-min(ylim_raw)) + \
                min(ylim_raw)*np.ones_like(ephem_times), c='red', linestyle='-',
                marker='^', markerfacecolor='red', markeredgecolor='black', ms=3,
                lw=0, zorder=3, rasterized=True)

    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time
    p = allq['inj_model'] if inj else np.nan
    injdepth = (p['params'].rp)**2 if inj else np.nan

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.01, max_time+timelen*0.01
    ax_raw.set(xlabel='', ylabel='',
        xlim=[xmin,xmax],
        ylim = ylim_raw)
    ax_raw.set_title(
        'KIC:{:s}, {:s}, q_flag>0, KEBC_P: {:.4f}. '.format(
        str(kicid), ap.upper(), kebc_period)+\
        'day, inj={:s}, dtr, iter whitened. '.format(str(inj))+\
        'Depth_inj: {:.4g}'.format(injdepth),
        fontsize='xx-small')
    ax_raw.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)

    ############################
    # PHASE-FOLDED LIGHTCURVES #
    ############################
    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, np.nanmax(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]

    for ix, ax in enumerate(axs_φ):

        if ix == len(nbestperiods):
            break
        foldperiod = nbestperiods[ix]

        # Recover quantities to plot, defined on φ=[0,1]
        fbls = allq['dipfind']['bls'][ap]['finebls'][foldperiod]
        plotφ = fbls['φ']
        plotfluxs = fbls['flux_φ']
        binplotφ = fbls['binned_φ']
        binplotfluxs = fbls['binned_flux_φ']
        φ_0 = fbls['φ_0']
        if ix == 0:
            bestφ_0 = φ_0
        # Want to wrap over φ=[-2,2]
        plotφ = np.concatenate(
                (plotφ-2.,plotφ-1.,plotφ,plotφ+1.,plotφ+2.)
                )
        plotfluxs = np.concatenate(
                (plotfluxs,plotfluxs,plotfluxs,plotfluxs,plotfluxs)
                )
        binplotφ = np.concatenate(
                (binplotφ-2.,binplotφ-1.,binplotφ,binplotφ+1.,binplotφ+2.)
                )
        binplotfluxs = np.concatenate(
                (binplotfluxs,binplotfluxs,binplotfluxs,binplotfluxs,binplotfluxs)
                )

        # Make the phased LC plot
        ax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=3,alpha=1,color='gray',
                rasterized=True)

        # Overlay the binned phased LC plot
        if phasebin:
            ax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=15,color='blue',
                    rasterized=True)

        # Compute the BLS model
        φ_ing, φ_egr = fbls['φ_ing'], fbls['φ_egr']
        δ = fbls['serialdict']['blsresult']['transdepth']

        def _get_bls_model_flux(φ, φ_0, φ_ing, φ_egr, δ, median_flux):
            flux = np.zeros_like(φ)
            for ix, phase in enumerate(φ-φ_0):
                if phase < φ_ing - φ_0 or phase > φ_egr - φ_0:
                    flux[ix] = median_flux
                else:
                    flux[ix] = median_flux - δ
            return flux

        # Overplot the BLS model
        bls_flux = _get_bls_model_flux(np.arange(-2,2,0.005), φ_0, φ_ing,
                φ_egr, δ, np.median(plotfluxs))
        ax.step(np.arange(-2,2,0.005)-φ_0, bls_flux, where='mid',
                c='red', ls='-')

        # Overlay the inset plot
        if inset:
            # Bottom left
            subpos = [0.03,0.03,0.25,0.2]
            i1ax = irp._add_inset_axes(ax,f,subpos)

            i1ax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=2*0.1,color='gray',
                    rasterized=True)
            if phasebin:
                i1ax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=10*0.2,
                        color='blue', zorder=20, rasterized=True)
            i1ymin = np.mean(plotfluxs) - 3.5*np.std(plotfluxs)
            i1ymax = np.mean(plotfluxs) + 2*np.std(plotfluxs)
            i1ax.set_ylim([i1ymin, i1ymax])
            i1axylim = i1ax.get_ylim()

            i1ax.set(ylabel='', xlim=[-0.7,0.7])
            i1ax.get_xaxis().set_visible(False)
            i1ax.get_yaxis().set_visible(False)

            # Bottom right
            subpos = [0.72,0.03,0.25,0.2]
            i2ax = irp._add_inset_axes(ax,f,subpos)

            i2ax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=2*0.1,color='gray',
                    rasterized=True)
            if phasebin:
                i2ax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=10*0.2,
                        color='blue', zorder=20, rasterized=True)
            i2ymin = np.mean(plotfluxs) - 3.5*np.std(plotfluxs)
            i2ymax = np.mean(plotfluxs) + 2*np.std(plotfluxs)
            i2ax.set_ylim([i2ymin, i2ymax])
            i2axylim = i2ax.get_ylim()

            i2ax.set(ylabel='', xlim=[-0.6,-0.4])
            i2ax.get_xaxis().set_visible(False)
            i2ax.get_yaxis().set_visible(False)


        t0 = min_time + φ_0*foldperiod
        φ_dur = φ_egr - φ_ing if φ_egr > φ_ing else (1+φ_egr) - φ_ing
        T_dur = foldperiod * φ_dur

        txt = 'P_fold: {:.2f} d, P_fold/P_EB: {:.3f}, T_dur: {:.1f} hr, t_0: {:.1f}'.format(
                foldperiod, foldperiod/kebc_period, T_dur*24., t0)
        t = ax.text(0.5, 0.98, txt, horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes, fontsize='xx-small')
        t.set_bbox(dict(facecolor='white', alpha=1., edgecolor='black'))


        ax.set(ylabel='', xlim=[-0.1,0.1])
        ymin = np.mean(plotfluxs) - 3*np.std(plotfluxs)
        ymax = np.mean(plotfluxs) + 1.5*np.std(plotfluxs)
        ax.set_ylim([ymin, ymax])
        axylim = ax.get_ylim()
        ax.vlines([0.],np.nanmin(axylim),np.nanmin(axylim)+0.05*(np.nanmax(axylim)-np.nanmin(axylim)),
                colors='red', linestyles='-', alpha=0.9, zorder=-1)
        ax.set_ylim([ymin, ymax])
        if inset:
            i1ax.vlines([-0.5,0.5], np.nanmin(i1axylim), np.nanmax(i1axylim), colors='black',
                    linestyles='-', alpha=0.7, zorder=30)
            i2ax.vlines([-0.5,0.5], np.nanmin(i2axylim), np.nanmax(i2axylim), colors='black',
                    linestyles='-', alpha=0.7, zorder=30)
            i1ax.vlines([0.], np.nanmin(i1axylim),
                    np.nanmin(i1axylim)+0.05*(np.nanmax(i1axylim)-np.nanmin(i1axylim)),
                    colors='red', linestyles='-', alpha=0.9, zorder=-1)
            i1ax.set_ylim([i1ymin, i1ymax])
        if ix in [0,1,2]:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if ix in [1,2,4]:
            ax.set_yticks([])
            ax.set_yticklabels([])




    ################
    # PERIODOGRAMS #
    ################
    pgdc = allq['dipfind']['bls'][ap]['coarsebls']
    pgdf = allq['dipfind']['bls'][ap]['finebls']

    ax_pg.plot(pgdc['periods'], pgdc['lspvals'], 'k-', zorder=10)

    pwr_ylim = ax_pg.get_ylim()

    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, np.nanmax(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]
    cbestperiod = nbestperiods[0]

    # We want the FINE best period, not the coarse one (although the frequency
    # grid might actually be small enough that this doesn't improve much!)
    fbestperiod = pgdf[cbestperiod]['serialdict']['bestperiod']

    best_t0 = min_time + bestφ_0*fbestperiod
    # Show 5 _selected_ periods in red.
    ax_pg.vlines(nbestperiods, np.nanmin(pwr_ylim), np.nanmax(pwr_ylim), colors='r',
            linestyles='solid', alpha=1, lw=1, zorder=-5)
    # Underplot 5 best coarse periods. (Shows negative cases).
    ax_pg.vlines(pgdc['nbestperiods'][:6], np.nanmin(pwr_ylim), np.nanmax(pwr_ylim),
            colors='gray', linestyles='dotted', lw=1, alpha=0.7, zorder=-10)

    p = allq['inj_model'] if inj else np.nan
    injperiod = p['params'].per if inj else np.nan
    inj_t0 = p['params'].t0 if inj else np.nan
    if inj:
        ax_pg.vlines(injperiod, np.nanmin(pwr_ylim), np.nanmax(pwr_ylim), colors='g',
                linestyles='-', alpha=0.8, zorder=10)

    if inj:
        txt = 'P_inj: %.4f d\nP_rec: %.4f d\nt_0,inj: %.4f\nt_0,rec: %.4f' % \
              (injperiod, fbestperiod, inj_t0, best_t0)
    else:
        txt = 'P_rec: %.4f d\nt_0,rec: %.4f' % \
              (fbestperiod, best_t0)

    ax_pg.text(0.96,0.96,txt,horizontalalignment='right',
            verticalalignment='top',
            transform=ax_pg.transAxes, fontsize='x-small')

    ax_pg.set(xlabel='', xscale='log')
    ax_pg.get_yaxis().set_ticks([])
    ax_pg.set(ylabel='')
    ax_pg.set(ylim=[np.nanmin(pwr_ylim),np.nanmax(pwr_ylim)])
    ax_pg.set(xlim=[np.nanmin(pgdc['periods']),np.nanmax(pgdc['periods'])])

    # Figure out names and write.
    savedir = '../results/vet_bls/'
    plotname = str(kicid)+'.png'

    f.savefig(savedir+plotname, dpi=300, bbox_inches='tight')

    LOGINFO('Made & saved vet bls plot to {:s}'.format(savedir+plotname))


if __name__ == '__main__':

    # First, be sure to run:
    # >>> python run_the_machine.py --pkltocsv --inj 0

    # Then from ipython:
    # >>> import injrecovresult_analysis as irra
    # >>> irra.summarize_realsearch_result()
    # then you'll have candidates_sort_allN.csv

    # This throws out obvious harmonics, and keeps only SNR_pf > 3 candidates.
    # (No real BLS detection threshold, which is probably bad)
    df = pd.read_csv('../results/real_search/candidates_sort_allN.csv')
    ok_ids = np.array(df['kicid'])

    # Now get pickles
    pkldir = '/media/luke/LGB_tess_data/cbp_data_170705_realsearch/'
    stage = 'realsearch_real'
    pklnames = os.listdir(pkldir)
    lcdnames = [pn for pn in pklnames if 'allq' not in pn and
            int(pn.split('_')[0]) in ok_ids]
    allqnames = [pn for pn in pklnames if 'allq' in pn and
            int(pn.split('_')[0]) in ok_ids]

    for lcdname in np.sort(lcdnames)[4::5]:
        kicid = lcdname.split('_')[0]

        lcd, loadfailed = ir.load_lightcurve_data( kicid, stage=stage,
                pkldir=pkldir)
        if loadfailed:
            continue

        allq, loadfailed = ir.load_allq_data(kicid, stage=stage, pkldir=pkldir)
        if loadfailed:
            continue

        # Finally, vet the BLS results w/ improved dipsearchplot
        vet_bls(lcd, allq, ap='sap', inj=False, varepoch='bls',
                phasebin=0.002, inset=True)
